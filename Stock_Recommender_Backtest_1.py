from src.backtest_portfolio import backtest_topk_portfolio_rollover

result = backtest_topk_portfolio_rollover(initial_cash=1_000_000)

print("\n====== 回测结果 ======")
print("期末资金：", f"{result['final_equity']:,.2f}")
print("总收益：", f"{result['total_pnl']:+,.2f}", f"({result['total_return']:.2%})")
print("年化收益率：", f"{result['annual_return']:.2%}")
print("最大回撤：", f"{result['max_drawdown']:.2%}")

result["details"].to_csv("output/portfolio_rollover_daily_details.csv", index=False, encoding="utf-8-sig")
print("✅ 明细已保存：output/portfolio_rollover_daily_details.csv")
