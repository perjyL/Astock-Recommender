from src.data_loader import get_index_constituents, get_stock_history
from src.feature_engineering import add_features
from src.model import train_model_cls
from src.predictor import make_decision
from src.visualization import plot_price_ma, plot_macd, plot_volume
from src.config import INDEX_CODE, START_DATE, END_DATE, USE_JOINT_TRANSFORMER, USE_JOINT_FINETUNE
from datetime import datetime
from src.recommender import hs300_recommendation
import os


def _is_end_date_today():
    for fmt in ("%Y%m%d", "%Y-%m-%d"):
        try:
            return datetime.strptime(END_DATE, fmt).date() == datetime.now().date()
        except ValueError:
            continue
    return False


def _get_feature_cols():
    try:
        from src.config import FEATURE_COLS
        return FEATURE_COLS
    except Exception:
        return [
            "MA5", "MA10", "MA20",
            "MACD", "DIF", "DEA",
            "VOL_MA5", "Volatility"
        ]


def _get_model_type_cls():
    try:
        from src.config import MODEL_TYPE_CLS
        return MODEL_TYPE_CLS
    except Exception:
        from src.config import MODEL_TYPE
        return MODEL_TYPE


def main():
    stocks = get_index_constituents(INDEX_CODE)
    if not stocks:
        print(f"指数 {INDEX_CODE} 成分股为空")
        return

    print(f"指数 {INDEX_CODE} 成分股数量：{len(stocks)}")

    # 示例演示
    symbol = stocks[0]
    print(f"\n分析股票：{symbol}")

    df = get_stock_history(symbol, use_realtime=_is_end_date_today())
    df = add_features(df)

    features = _get_feature_cols()

    df_train = df[features + ["Target"]].dropna()
    if df_train.empty:
        print("可用训练样本为空，跳过预测")
        return

    model_type = _get_model_type_cls()
    model = train_model_cls(df_train[features], df_train["Target"], model_type=model_type)

    df_features = df[features].dropna()
    if df_features.empty:
        print("最新特征为空，无法预测")
        return
    X_latest = df_features.iloc[[-1]]
    prob, signal = make_decision(model, X_latest)

    print(f"预测下一交易日上涨概率：{prob:.2f}")
    print(f"投资建议：{signal}")

    # 可视化
    plot_price_ma(df, symbol)
    plot_macd(df, symbol)
    plot_volume(df, symbol)



if __name__ == "__main__":

    use_realtime = _is_end_date_today()
    df = hs300_recommendation(use_realtime=use_realtime)
    if df is None or df.empty:
        print("推荐结果为空")
        raise SystemExit(0)
    os.makedirs("output", exist_ok=True)
    df.to_csv("output/hs300_recommendation.csv", index=False, encoding="utf-8-sig")
    print("\n预测结果已保存为：hs300_recommendation.csv")

    print("\n===== 沪深300 推荐结果（Top 10） =====")
    print(df.head(10))

    model_type = _get_model_type_cls()
    model_desc = model_type
    if model_type == "transformer" and USE_JOINT_TRANSFORMER:
        model_desc = "transformer + joint"
        if USE_JOINT_FINETUNE:
            model_desc += " + finetune"

    print(f"\n时间范围: {START_DATE} - {END_DATE} | 模型: {model_desc}")
    if use_realtime:
        print("已使用当日实时价格进行模拟运行")
