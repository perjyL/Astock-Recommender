import copy
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Ridge

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.config import (
    RANDOM_STATE,
    TRANSFORMER_WINDOW,
    TRANSFORMER_EPOCHS,
    MODEL_TYPE_CLS,
    MODEL_TYPE_REG,
)


# =========================================================
# 分类模型
# =========================================================
def train_rf_cls(X, y):
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    model.fit(X, y)
    return model


def train_xgb_cls(X, y):
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise ImportError("未安装 xgboost（分类），请先执行: pip install xgboost") from exc

    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        tree_method="hist"
    )
    model.fit(X, y)
    return model


# =========================================================
# 回归模型（Top-K组合核心）
# =========================================================
def train_ridge_reg(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = Ridge(alpha=1.0, random_state=RANDOM_STATE)
    model.fit(X_scaled, y)

    model.scaler = scaler
    return model


def train_rf_reg(X, y):
    model = RandomForestRegressor(
        n_estimators=400,
        max_depth=10,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    model.fit(X, y)
    return model


def train_xgb_reg(X, y):
    try:
        from xgboost import XGBRegressor
    except ImportError as exc:
        raise ImportError("未安装 xgboost（回归），请先执行: pip install xgboost") from exc

    model = XGBRegressor(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        tree_method="hist"
    )
    model.fit(X, y)
    return model


# =========================================================
# Transformer 分类
# =========================================================
class TransformerClassifier(torch.nn.Module):
    def __init__(self, input_dim, window, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.window = window
        self.input_dim = input_dim

        self.embedding = torch.nn.Linear(input_dim, d_model)
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True
        )
        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = torch.nn.Linear(d_model, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x)
        x = self.encoder(x)
        x = x[:, -1, :]
        x = self.fc(x)
        return self.sigmoid(x).squeeze()


def train_transformer(X, y, window=20, epochs=5):
    if len(X) <= window + 5:
        raise ValueError("样本长度不足以训练 Transformer")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_seq, y_seq = [], []
    y_values = y.to_numpy() if hasattr(y, "to_numpy") else np.asarray(y)

    for i in range(window, len(X_scaled)):
        X_seq.append(X_scaled[i - window:i])
        y_seq.append(y_values[i])

    X_seq = torch.tensor(np.array(X_seq), dtype=torch.float32)
    y_seq = torch.tensor(np.array(y_seq), dtype=torch.float32)

    model = TransformerClassifier(input_dim=X.shape[1], window=window)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCELoss()

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        output = model(X_seq).squeeze()
        loss = criterion(output, y_seq)
        loss.backward()
        optimizer.step()

    model.scaler = scaler
    model.window = window
    model.eval()
    return model


def train_transformer_joint(
    all_dfs,
    feature_cols,
    window=TRANSFORMER_WINDOW,
    epochs=TRANSFORMER_EPOCHS,
    batch_size=64,
    lr=1e-3
):
    print(f"\n📊 联合 Transformer 训练样本构建中...")

    X_all, y_all = [], []

    for df in all_dfs:
        X = df[feature_cols].values
        y = df["Target"].values
        for i in range(window, len(X)):
            X_all.append(X[i - window:i])
            y_all.append(y[i])

    if not X_all:
        raise ValueError("联合训练样本为空")

    X_all = np.array(X_all)
    y_all = np.array(y_all)

    N, T, F = X_all.shape
    scaler = StandardScaler()
    X_all_scaled = scaler.fit_transform(X_all.reshape(-1, F)).reshape(N, T, F)

    X_all = torch.tensor(X_all_scaled, dtype=torch.float32)
    y_all = torch.tensor(y_all, dtype=torch.float32)

    print(f"✅ 样本构建完成 | 样本数={len(X_all)} Window={window} 特征数={F}")

    dataset = TensorDataset(X_all, y_all)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = TransformerClassifier(input_dim=F, window=window)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch [{epoch}/{epochs}]", leave=True)

        for X_batch, y_batch in pbar:
            optimizer.zero_grad()
            preds = model(X_batch).squeeze()
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        print(f"✅ Epoch {epoch} 完成 | Avg Loss: {epoch_loss / len(dataloader):.4f}\n")

    model.scaler = scaler
    model.window = window
    model.feature_cols = feature_cols
    model.eval()
    return model


def finetune_transformer(
    base_model,
    X,
    y,
    window=TRANSFORMER_WINDOW,
    epochs=1,
    batch_size=64,
    lr=1e-4
):
    if epochs <= 0:
        return base_model

    if len(X) <= window + 5:
        raise ValueError("样本长度不足以微调 Transformer")

    X_values = X.to_numpy() if hasattr(X, "to_numpy") else np.asarray(X)
    y_values = y.to_numpy() if hasattr(y, "to_numpy") else np.asarray(y)

    scaler = base_model.scaler
    X_scaled = scaler.transform(X_values)

    X_seq, y_seq = [], []
    for i in range(window, len(X_scaled)):
        X_seq.append(X_scaled[i - window:i])
        y_seq.append(y_values[i])

    if not X_seq:
        raise ValueError("微调样本为空")

    X_seq = torch.tensor(np.array(X_seq), dtype=torch.float32)
    y_seq = torch.tensor(np.array(y_seq), dtype=torch.float32)

    model = copy.deepcopy(base_model)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    dataset = TensorDataset(X_seq, y_seq)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for _ in range(epochs):
        model.train()
        for X_batch, y_batch in dataloader:
            optimizer.zero_grad()
            preds = model(X_batch).squeeze()
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()

    model.scaler = scaler
    model.window = window
    model.eval()
    return model


# =========================================================
# ✅ 统一入口：分类 / 回归 分开
# =========================================================
def train_model_cls(X, y):
    if MODEL_TYPE_CLS == "randomforest":
        return train_rf_cls(X, y)
    elif MODEL_TYPE_CLS == "xgboost":
        return train_xgb_cls(X, y)
    elif MODEL_TYPE_CLS == "transformer":
        return train_transformer(X, y, window=TRANSFORMER_WINDOW, epochs=TRANSFORMER_EPOCHS)
    else:
        raise ValueError(f"未知 MODEL_TYPE_CLS: {MODEL_TYPE_CLS}")


def train_model_reg(X, y):
    if MODEL_TYPE_REG == "ridge":
        return train_ridge_reg(X, y)
    elif MODEL_TYPE_REG == "rf_reg":
        return train_rf_reg(X, y)
    elif MODEL_TYPE_REG == "xgb_reg":
        return train_xgb_reg(X, y)
    else:
        raise ValueError(f"未知 MODEL_TYPE_REG: {MODEL_TYPE_REG}")


# =========================================================
# 兼容旧代码：如果还有地方调用 train_model(X,y)
# 默认按“分类”训练，避免你旧 backtest.py/演示炸掉
# =========================================================
def train_model(X, y):
    return train_model_cls(X, y)
