from src.backtest_portfolio import backtest_topk_portfolio

print("✅ 进入 Stock_Recommender_Backtest_1.py")

result = backtest_topk_portfolio()

print("\n====== 回测结果 ======")
print("年化收益率：", f"{result['annual_return']:.2%}")
print("最大回撤：", f"{result['max_drawdown']:.2%}")

# 保存每日明细（包含TopK、当日收益、净值）
result["details"].to_csv("output/sz50_portfolio_topk_daily_details.csv", encoding="utf-8-sig")
print("✅ 已保存：output/portfolio_topk_daily_details.csv")
