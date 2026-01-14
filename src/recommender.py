import pandas as pd

try:
    import torch
except ImportError:
    torch = None

from src.data_loader import (
    get_stock_history,
    get_index_constituents,
    get_index_constituents_with_name,
)
from src.feature_engineering import add_features
from src.model import train_model_cls, train_transformer_joint, finetune_transformer
from src.config import (
    INDEX_CODE,
    MODEL_TYPE_CLS,
    USE_JOINT_TRANSFORMER,
    USE_JOINT_FINETUNE,
    JOINT_FINETUNE_EPOCHS,
    JOINT_FINETUNE_LR,
    BUY_THRESHOLD,
    SELL_THRESHOLD,
)


# ======================================================
# 全局联合 Transformer（只训练一次）
# ======================================================
JOINT_TRANSFORMER_MODEL = None


# ======================================================
# 投资建议规则
# ======================================================
def get_recommendation(prob):
    if prob >= BUY_THRESHOLD:
        return "Buy"
    elif prob >= SELL_THRESHOLD:
        return "Hold"
    else:
        return "Sell"


DEFAULT_FEATURES = [
    "MA5", "MA10", "MA20",
    "DIF", "DEA", "MACD",
    "VOL_MA5", "Volatility"
]


def _get_feature_cols():
    try:
        from src.config import FEATURE_COLS
        return FEATURE_COLS
    except Exception:
        return DEFAULT_FEATURES


def _get_model_type_cls():
    try:
        from src.config import MODEL_TYPE_CLS as mt
        return mt
    except Exception:
        return MODEL_TYPE_CLS


# ======================================================
# Transformer 专用预测函数
# ======================================================
def transformer_predict(model, X, feature_cols=None):
    """
    使用最后 window 天数据做 Transformer 预测
    """
    if torch is None:
        raise ImportError("未安装 torch，Transformer 预测不可用")

    if feature_cols is None:
        feature_cols = getattr(model, "feature_cols", None)

    if feature_cols is not None and hasattr(X, "columns"):
        X = X[feature_cols]

    X_values = X.to_numpy() if hasattr(X, "to_numpy") else X

    if len(X_values) < model.window:
        return None

    if not hasattr(model, "scaler"):
        raise RuntimeError("Transformer 模型缺少 scaler，无法做归一化预测")

    X_scaled = model.scaler.transform(X_values)

    seq = torch.tensor(
        X_scaled[-model.window:],
        dtype=torch.float32
    ).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        out = model(seq)

    # 兼容 sigmoid / softmax 两种输出
    if out.ndim == 2 and out.shape[1] == 2:
        prob = torch.softmax(out, dim=1)[0, 1].item()
    else:
        prob = out.squeeze().item()

    return float(prob)


# ======================================================
# 沪深300 推荐主函数
# ======================================================
def hs300_recommendation(use_realtime=False):
    global JOINT_TRANSFORMER_MODEL

    name_map = get_index_constituents_with_name(INDEX_CODE)
    symbols = list(name_map.keys()) if name_map else get_index_constituents(INDEX_CODE)
    if not symbols:
        return pd.DataFrame()
    features = _get_feature_cols()
    model_type = (_get_model_type_cls() or "").lower()

    results = []
    use_joint = bool(USE_JOINT_TRANSFORMER)

    # ==================================================
    # 🚀 联合训练 Transformer（只执行一次）
    # ==================================================
    if model_type == "transformer" and use_joint:
        if JOINT_TRANSFORMER_MODEL is None:
            print("🚀 开始联合训练 Transformer（指数成分股横截面 + 时间）...")

            all_dfs = []
            for code in symbols:
                try:
                    df_i = get_stock_history(code, use_realtime=use_realtime)
                    if df_i is None or df_i.empty:
                        continue
                    df_i = add_features(df_i)
                    df_i = df_i[features + ["Target"]].dropna()
                    if len(df_i) >= 30:
                        all_dfs.append(df_i)
                except Exception:
                    continue

            try:
                JOINT_TRANSFORMER_MODEL = train_transformer_joint(
                    all_dfs,
                    feature_cols=features
                )
                print("✅ 联合 Transformer 训练完成")
            except Exception as exc:
                print(f"⚠️ 联合 Transformer 训练失败：{repr(exc)}，将回退为逐股训练")
                JOINT_TRANSFORMER_MODEL = None
                use_joint = False

    # ==================================================
    # 📊 逐股票预测
    # ==================================================
    for code in symbols:
        name = name_map.get(code, code)

        try:
            # 1️⃣ 数据加载
            df_raw = get_stock_history(code, use_realtime=use_realtime)
            if df_raw.empty:
                raise ValueError("行情数据为空")

            last_date = df_raw.index[-1]
            last_close = df_raw["收盘"].iloc[-1]

            df = add_features(df_raw)

            df_train = df[features + ["Target"]].dropna()
            df_features = df[features].dropna()

            if len(df_train) < 30 or len(df_features) == 0:
                raise ValueError("样本过短")

            # 2️⃣ 模型训练（非联合 Transformer）
            if model_type != "transformer" or not use_joint:
                model = train_model_cls(df_train[features], df_train["Target"], model_type=model_type)

            # 3️⃣ === 预测 ===
            if model_type == "transformer":
                model_use = JOINT_TRANSFORMER_MODEL if use_joint else model

                if use_joint and USE_JOINT_FINETUNE and JOINT_FINETUNE_EPOCHS > 0:
                    try:
                        model_use = finetune_transformer(
                            model_use,
                            df_train[features],
                            df_train["Target"],
                            window=model_use.window,
                            epochs=JOINT_FINETUNE_EPOCHS,
                            lr=JOINT_FINETUNE_LR
                        )
                    except ValueError:
                        pass

                prob = transformer_predict(model_use, df_features, feature_cols=features)
                if prob is None:
                    raise ValueError("Transformer 数据不足")

            else:
                prob = model.predict_proba(df_features.iloc[[-1]])[0, 1]

            # 4️⃣ 投资建议
            rec = get_recommendation(prob)

            results.append({
                "Code": code,
                "Name": name,
                "Last_Date": last_date.strftime("%Y-%m-%d"),
                "Last_Close": round(float(last_close), 2),
                "Up_Prob": round(prob, 4),
                "Recommendation": rec
            })

            print(f"{code} {name} → {rec} ({prob:.2f})")

        except Exception as e:
            # 🔴 现在会打印真实错误，方便你调试
            print(f"{code} {name} 数据异常：{repr(e)}")
            continue

    df_result = pd.DataFrame(results)
    if df_result.empty:
        return df_result
    df_result = df_result.sort_values("Up_Prob", ascending=False)
    df_result.insert(0, "Rank", range(1, len(df_result) + 1))

    return df_result
