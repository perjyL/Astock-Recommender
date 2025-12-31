from src.data_loader import get_index_constituents, get_stock_history
from src.feature_engineering import add_features
from src.model import train_model
from src.predictor import make_decision
from src.visualization import plot_price_ma, plot_macd, plot_volume
from src.config import INDEX_CODE
from src.recommender import hs300_recommendation
import pandas as pd


def main():
    stocks = get_index_constituents(INDEX_CODE)

    print(f"指数 {INDEX_CODE} 成分股数量：{len(stocks)}")

    # 示例演示
    symbol = stocks[0]
    print(f"\n分析股票：{symbol}")

    df = get_stock_history(symbol)
    df = add_features(df)

    features = [
        "MA5", "MA10", "MA20",
        "MACD", "DIF", "DEA",
        "VOL_MA5", "Volatility"
    ]

    X = df[features]
    y = df["Target"]

    model = train_model(X[:-1], y[:-1])

    prob, signal = make_decision(model, X.iloc[[-1]])

    print(f"预测下一交易日上涨概率：{prob:.2f}")
    print(f"投资建议：{signal}")

    # 可视化
    plot_price_ma(df, symbol)
    plot_macd(df, symbol)
    plot_volume(df, symbol)



if __name__ == "__main__":
    df = hs300_recommendation()
    df.to_csv("output/hs300_recommendation.csv", index=False, encoding="utf-8-sig")

    print("\n===== 沪深300 推荐结果（Top 10） =====")
    print(df.head(10))
