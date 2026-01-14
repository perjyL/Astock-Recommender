# src/backtest.py
import time
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, brier_score_loss
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

from src.data_loader import get_stock_history, get_index_constituents
from src.feature_engineering import add_features
from src.config import INDEX_CODE, MODEL_TYPE_CLS

# 默认特征列（若你未来在 config.py 增加 FEATURE_COLS，这里也会自动用）
DEFAULT_FEATURES = [
    "MA5", "MA10", "MA20",
    "MACD", "DIF", "DEA",
    "VOL_MA5", "Volatility"
]

def _get_feature_cols():
    try:
        from src.config import FEATURE_COLS
        return FEATURE_COLS
    except Exception:
        return DEFAULT_FEATURES


def prob_to_signal(prob):
    if prob >= 0.6:
        return "Buy"
    elif prob >= 0.4:
        return "Hold"
    else:
        return "Sell"


def _train_model_safe(X_train, y_train, model_type):
    """兼容 train_model(X,y) 或 train_model(X,y,model_type) 两种签名"""
    from src.model import train_model_cls
    return train_model_cls(X_train, y_train, model_type=model_type)


def _predict_prob_safe(model, X_test, model_type, X_hist_for_transformer=None):
    """统一得到上涨概率"""
    mt = (model_type or "").lower()

    if "transformer" in mt:
        # transformer 没有 predict_proba，用你以前的 transformer_predict 逻辑
        from src.recommender import transformer_predict
        try:
            prob = transformer_predict(model, X_hist_for_transformer)
            return prob
        except Exception:
            return None

    # sklearn/xgb 分类模型
    if hasattr(model, "predict_proba"):
        return float(model.predict_proba(X_test)[0, 1])

    # 兜底：用 predict 输出(0/1)当概率
    if hasattr(model, "predict"):
        return float(model.predict(X_test)[0])

    raise RuntimeError("模型没有可用的 predict_proba/predict 接口")


def backtest_single_stock(
    symbol: str,
    start_test_date="2025-01-01",
    end_test_date="2025-12-31",
    min_train_size=200,
    verbose=True
):
    features = _get_feature_cols()

    df = get_stock_history(symbol)
    if df is None or df.empty:
        return pd.DataFrame()
    df = add_features(df)
    test_df = df.loc[start_test_date:end_test_date]

    records = []
    if verbose:
        print(f"\n========== 股票 {symbol} ==========")

    for date in test_df.index:
        # walk-forward：预测当天时，只用到 date 之前的数据（不含当天未来标签）
        train_df = df.loc[:date].iloc[:-1]
        train_df = train_df[features + ["Target"]].dropna()
        if len(train_df) < min_train_size:
            continue

        X_train = train_df[features]
        y_train = train_df["Target"]
        model = _train_model_safe(X_train, y_train, MODEL_TYPE_CLS)

        X_test = df.loc[[date], features].dropna()
        if X_test.empty:
            continue

        prob_up = _predict_prob_safe(
            model, X_test, MODEL_TYPE_CLS, X_hist_for_transformer=train_df[features]
        )
        if prob_up is None or (isinstance(prob_up, float) and np.isnan(prob_up)):
            continue

        pred = int(prob_up >= 0.5)
        signal = prob_to_signal(prob_up)
        true = int(df.loc[date, "Target"])
        correct = "✓" if pred == true else "✗"

        if verbose:
            print(f"[{date.date()}] train={len(train_df)} prob={prob_up:.3f} "
                  f"signal={signal} true={'上涨' if true==1 else '下跌'} {correct}")

        records.append({
            "date": date,
            "symbol": symbol,
            "prob_up": float(prob_up),
            "signal": signal,
            "y_true": true,
            "y_pred": pred
        })

    return pd.DataFrame(records)


def backtest_hs300_2025(verbose=True):
    symbols = get_index_constituents(INDEX_CODE)
    print(f"\n开始指数 {INDEX_CODE} 回测（股票数量：{len(symbols)}）")
    print("=" * 60)

    all_results = []
    for i, symbol in enumerate(symbols, 1):
        print(f"\n>>> [{i}/{len(symbols)}] 回测股票 {symbol}")
        try:
            df = backtest_single_stock(symbol, verbose=verbose)
            if not df.empty:
                all_results.append(df)
        except Exception as e:
            print(f"股票 {symbol} 回测失败：{e}")

    if not all_results:
        return None

    return pd.concat(all_results, ignore_index=True)


def evaluate_overall(result_df: pd.DataFrame):
    acc = accuracy_score(result_df["y_true"], result_df["y_pred"])
    cm = confusion_matrix(result_df["y_true"], result_df["y_pred"])
    brier = brier_score_loss(result_df["y_true"], result_df["prob_up"])

    print("\n========== 整体回测结果 ==========")
    print(f"总体方向预测准确率：{acc:.4f}")
    print("混淆矩阵（真实 x 预测）：")
    print(cm)
    print(f"Brier Score（概率误差）：{brier:.4f}")

    return acc, cm, brier
