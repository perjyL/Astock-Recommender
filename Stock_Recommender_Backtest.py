"""
Stock_Recommender.py
-------------------
主程序入口：

1. 单只股票当日推荐（演示）
2. 全股票 2025 年逐日滚动回测
3. 实时打印预测与真实结果
4. 汇总整体模型准确率与误差指标
"""

from src.data_loader import get_index_constituents, get_stock_history
from src.feature_engineering import add_features
from src.model import train_model
from src.predictor import make_decision
from src.visualization import plot_price_ma, plot_macd, plot_volume
from src.config import INDEX_CODE

from src.backtest import (
    backtest_hs300_2025,
    evaluate_overall
)

import pandas as pd


# =========================
# 1. 单只股票推荐演示
# =========================
def main():
    print("\n===== 单只股票推荐演示（盘后） =====")

    stocks = get_index_constituents(INDEX_CODE)
    print(f"指数 {INDEX_CODE} 成分股数量：{len(stocks)}")

    # 作业展示：只取第一只
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

    # 用历史数据训练（不包含最后一天）
    model = train_model(X[:-1], y[:-1])

    # 预测最后一天对应的“下一交易日”
    prob, signal = make_decision(model, X.iloc[[-1]])

    print(f"预测下一交易日上涨概率：{prob:.2f}")
    print(f"投资建议：{signal}")

    # 可视化（技术分析支撑）
    plot_price_ma(df, symbol)
    plot_macd(df, symbol)
    plot_volume(df, symbol)


# =========================
# 2. 沪深300 2025 年回测
# =========================
def run_hs300_backtest():
    print("\n===== 开始 2025 年 全股票回测 =====")
    print("说明：逐日 Walk-Forward，无未来数据泄露")
    print("=" * 70)

    # verbose=True：逐日打印每只股票的预测过程
    # 作业演示建议 True，完整跑建议 False
    results = backtest_hs300_2025(verbose=True)

    if results is None or results.empty:
        print("回测失败：未获得任何结果")
        return

    # 整体评估
    evaluate_overall(results)

    # 额外统计：Buy / Hold / Sell 分布
    print("\n===== 投资建议分布 =====")
    print(results["signal"].value_counts())

    # 保存回测结果（方便写报告/画图）
    results.to_csv("output/hs300_backtest_results.csv", index=False)
    print("\n回测结果已保存为：hs300_backtest_results.csv")


# =========================
# 程序入口
# =========================
if __name__ == "__main__":
    # ① 单只股票推荐（展示系统功能）
    main()

    # ② 全市场回测（验证模型是否合适）
    run_hs300_backtest()
