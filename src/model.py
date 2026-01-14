import copy
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Ridge

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

from src.config import (
    RANDOM_STATE,
    TRANSFORMER_WINDOW,
    TRANSFORMER_EPOCHS,
    MODEL_TYPE_CLS,
    MODEL_TYPE_REG,
)


def _require_torch():
    if torch is None or nn is None:
        raise ImportError("未安装 torch，Transformer 相关功能不可用，请先安装 torch")


def _filter_valid_xy(X, y):
    """清理 NaN/Inf，避免训练阶段报错。"""
    X_values = X.to_numpy() if hasattr(X, "to_numpy") else np.asarray(X)
    y_values = y.to_numpy() if hasattr(y, "to_numpy") else np.asarray(y)

    if len(X_values) != len(y_values):
        raise ValueError("X 与 y 长度不一致")

    if X_values.ndim == 1:
        X_values = X_values.reshape(-1, 1)

    mask = np.isfinite(X_values).all(axis=1) & np.isfinite(y_values)
    if mask.sum() == 0:
        raise ValueError("有效样本为空（可能全部为 NaN/Inf）")

    if hasattr(X, "loc"):
        X_clean = X.loc[mask]
    else:
        X_clean = X_values[mask]

    if hasattr(y, "loc"):
        y_clean = y.loc[mask]
    else:
        y_clean = y_values[mask]

    return X_clean, y_clean


# =========================================================
# 分类模型
# =========================================================
def train_rf_cls(X, y):
    X, y = _filter_valid_xy(X, y)
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

    X, y = _filter_valid_xy(X, y)
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
    X, y = _filter_valid_xy(X, y)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = Ridge(alpha=1.0, random_state=RANDOM_STATE)
    model.fit(X_scaled, y)

    model.scaler = scaler
    return model


def train_rf_reg(X, y):
    X, y = _filter_valid_xy(X, y)
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

    X, y = _filter_valid_xy(X, y)
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
if torch is not None and nn is not None:
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
else:
    class TransformerClassifier:
        def __init__(self, *args, **kwargs):
            raise ImportError("未安装 torch，TransformerClassifier 不可用")


def train_transformer(X, y, window=20, epochs=5):
    _require_torch()
    X, y = _filter_valid_xy(X, y)
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
    _require_torch()
    print(f"\n📊 联合 Transformer 训练样本构建中...")

    X_all, y_all = [], []

    for df in all_dfs:
        df_use = df[feature_cols + ["Target"]].dropna()
        if df_use.empty or len(df_use) <= window:
            continue
        X = df_use[feature_cols].values
        y = df_use["Target"].values
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
    _require_torch()
    if epochs <= 0:
        return base_model

    X, y = _filter_valid_xy(X, y)
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
def _normalize_cls_type(model_type: str):
    mt = (model_type or "").lower().strip()
    if mt in {"rf", "randomforest", "random_forest"}:
        return "randomforest"
    if mt in {"xgb", "xgboost"}:
        return "xgboost"
    if mt in {"transformer"}:
        return "transformer"
    return mt


def train_model_cls(X, y, model_type: str | None = None):
    mt = _normalize_cls_type(model_type or MODEL_TYPE_CLS)
    if mt == "randomforest":
        return train_rf_cls(X, y)
    elif mt == "xgboost":
        return train_xgb_cls(X, y)
    elif mt == "transformer":
        return train_transformer(X, y, window=TRANSFORMER_WINDOW, epochs=TRANSFORMER_EPOCHS)
    else:
        raise ValueError(f"未知 MODEL_TYPE_CLS: {mt}")


def train_model_reg(X, y, model_type: str | None = None):
    mt = (model_type or MODEL_TYPE_REG).lower().strip()
    if mt == "ridge":
        return train_ridge_reg(X, y)
    elif mt in {"rf_reg", "randomforest", "random_forest", "rf"}:
        return train_rf_reg(X, y)
    elif mt in {"xgb_reg", "xgboost", "xgb"}:
        return train_xgb_reg(X, y)
    else:
        raise ValueError(f"未知 MODEL_TYPE_REG: {mt}")


# =========================================================
# 兼容旧代码：如果还有地方调用 train_model(X,y)
# 默认按“分类”训练，避免你旧 backtest.py/演示炸掉
# =========================================================
def train_model(X, y, model_type: str | None = None):
    return train_model_cls(X, y, model_type=model_type)
