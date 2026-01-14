# src/visualization.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 设置中文字体（Windows）
plt.rcParams["font.sans-serif"] = ["SimHei"]  # 黑体
plt.rcParams["axes.unicode_minus"] = False    # 解决负号显示问题


# =========================
# 技术分析图
# =========================
def plot_price_ma(df, symbol):
    plt.figure(figsize=(12, 6))
    plt.plot(df["收盘"], label="Close")
    plt.plot(df["MA5"], label="MA5")
    plt.plot(df["MA10"], label="MA10")
    plt.plot(df["MA20"], label="MA20")
    plt.title(f"{symbol} 价格与均线")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_macd(df, symbol):
    plt.figure(figsize=(12, 4))
    plt.bar(df.index, df["MACD"], label="MACD")
    plt.plot(df["DIF"], label="DIF")
    plt.plot(df["DEA"], label="DEA")
    plt.title(f"{symbol} MACD 指标")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_volume(df, symbol):
    plt.figure(figsize=(12, 4))
    plt.bar(df.index, df["成交量"], label="Volume")
    plt.plot(df["VOL_MA5"], label="VOL_MA5")
    plt.title(f"{symbol} 成交量变化")
    plt.legend()
    plt.tight_layout()
    plt.show()


# =========================
# 回测汇报核心图表（新增）
# =========================
def _ensure_dir(path: str):
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def _save_or_show(save_path: str | None):
    if save_path:
        _ensure_dir(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def compute_drawdown(equity: pd.Series) -> pd.Series:
    equity = equity.dropna()
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return dd


def plot_equity_curve(details_df: pd.DataFrame,
                      benchmark_df: pd.DataFrame | None = None,
                      title="策略净值曲线（含基准）",
                      save_path: str | None = None):
    """
    details_df 需要至少包含:
      - date
      - equity  (策略累计净值)
    benchmark_df 如果传入，需要包含:
      - date 或 index 为日期
      - bench_equity (基准累计净值) 或 收盘
    """
    df = details_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")

    plt.figure(figsize=(12, 6))
    plt.plot(df["equity"], label="Strategy")

    if benchmark_df is not None and len(benchmark_df) > 0:
        b = benchmark_df.copy()

        # 兼容 index 或 date 列
        if "date" in b.columns:
            b["date"] = pd.to_datetime(b["date"])
            b = b.sort_values("date").set_index("date")

        if "bench_equity" in b.columns:
            bench_equity = b["bench_equity"]
        elif "收盘" in b.columns:
            # 用收盘构造基准净值（归一化从1开始）
            bench_equity = (1 + b["收盘"].pct_change().fillna(0)).cumprod()
        else:
            bench_equity = None

        if bench_equity is not None:
            # 对齐日期
            bench_equity = bench_equity.reindex(df.index).ffill()
            plt.plot(bench_equity, label="Benchmark")

    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Equity (Normalized)")
    plt.legend()
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    _save_or_show(save_path)


def plot_drawdown_curve(details_df: pd.DataFrame,
                        title="回撤曲线（Drawdown）",
                        save_path: str | None = None):
    df = details_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")

    dd = compute_drawdown(df["equity"])

    plt.figure(figsize=(12, 4))
    plt.plot(dd, label="Drawdown")
    plt.axhline(0, linewidth=1)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    _save_or_show(save_path)


def plot_return_hist(details_df: pd.DataFrame,
                     col="portfolio_ret",
                     bins=60,
                     title="收益分布直方图",
                     save_path: str | None = None):
    df = details_df.copy()
    r = df[col].dropna().astype(float)

    plt.figure(figsize=(10, 5))
    plt.hist(r, bins=bins)
    plt.title(title)
    plt.xlabel("Return")
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    _save_or_show(save_path)


def plot_pred_vs_true_scatter(pred_records_df: pd.DataFrame,
                              title="预测收益 vs 实际收益（散点图）",
                              save_path: str | None = None):
    """
    pred_records_df: 每日每股预测记录（你在 backtest_portfolio 里可以保存）
      - pred
      - true_ret
    """
    if pred_records_df is None or len(pred_records_df) == 0:
        print("⚠️ pred_records_df 为空，无法绘制预测 vs 实际散点图")
        return

    df = pred_records_df.copy()
    x = df["pred"].astype(float)
    y = df["true_ret"].astype(float)

    plt.figure(figsize=(7, 7))
    plt.scatter(x, y, s=8)
    plt.axhline(0, linewidth=1)
    plt.axvline(0, linewidth=1)
    plt.title(title)
    plt.xlabel("Predicted future return")
    plt.ylabel("Realized future return")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    _save_or_show(save_path)


def plot_cash_utilization(details_df: pd.DataFrame,
                          title="资金利用率（Invested / Total）",
                          save_path: str | None = None):
    """
    details_df 需要包含:
      - date
      - invested_ratio  (0~1)
    """
    if "invested_ratio" not in details_df.columns:
        print("⚠️ details_df 缺少 invested_ratio，无法绘制资金利用率")
        return

    df = details_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")

    plt.figure(figsize=(12, 4))
    plt.plot(df["invested_ratio"], label="Invested Ratio")
    plt.ylim(0, 1.05)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Ratio")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    _save_or_show(save_path)


def plot_turnover(details_df: pd.DataFrame,
                  title="换手率（Turnover）",
                  save_path: str | None = None):
    """
    details_df 需要包含:
      - date
      - turnover  (0~1 或更大，取决于定义)
    """
    if "turnover" not in details_df.columns:
        print("⚠️ details_df 缺少 turnover，无法绘制换手率")
        return

    df = details_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")

    plt.figure(figsize=(12, 4))
    plt.plot(df["turnover"], label="Turnover")
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Turnover")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    _save_or_show(save_path)


def plot_total_balance(details_df: pd.DataFrame,
                       title="每日总余额（Total Balance）",
                       save_path: str | None = None):
    """
    details_df 需要包含:
      - date
      - total_balance
    """
    if "total_balance" not in details_df.columns:
        print("⚠️ details_df 缺少 total_balance，无法绘制总余额曲线")
        return

    df = details_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")

    plt.figure(figsize=(12, 5))
    plt.plot(df["total_balance"], label="Total Balance")
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Balance")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    _save_or_show(save_path)


def plot_daily_pnl(details_df: pd.DataFrame,
                   title="每日盈亏（ΔBalance）",
                   save_path: str | None = None):
    """
    details_df 需要包含:
      - date
      - pnl  (当天余额变化额)
    """
    if "pnl" not in details_df.columns:
        print("⚠️ details_df 缺少 pnl，无法绘制每日盈亏")
        return

    df = details_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")

    plt.figure(figsize=(12, 4))
    plt.plot(df["pnl"], label="PnL")
    plt.axhline(0, linewidth=1)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("PnL")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    _save_or_show(save_path)
