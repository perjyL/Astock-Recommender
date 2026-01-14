# src/backtest_portfolio.py
import time
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.data_loader import get_index_constituents, get_stock_history
from src.feature_engineering import add_features
from src.config import INDEX_CODE, MODEL_TYPE_REG

# -----------------------------
# 默认参数（若 config 没写就用默认）
# -----------------------------
DEFAULT_FEATURES = ["MA5", "MA10", "MA20", "MACD", "DIF", "DEA", "VOL_MA5", "Volatility"]

DEFAULT_START = "2025-01-01"
DEFAULT_END = "2025-12-31"

DEFAULT_MIN_TRAIN = 200
DEFAULT_TOP_K = 20

# 你的 N：预测未来 N 天涨跌/收益
DEFAULT_HOLD_N = 5  # 例如 5 表示预测 ret_5d
DEFAULT_TARGET_COL = "ret_5d"

# 每天真实记账用：t -> t+1 的真实收益
DEFAULT_REALIZED_RET_COL = "ret_1d_fwd"

# 权重方式：equal / proportional / softmax
DEFAULT_WEIGHT_MODE = "equal"
DEFAULT_SOFTMAX_TAU = 1.0

# 成本/滑点：简单按“当日新开仓的桶”扣一次（单边），你也可以改成双边
DEFAULT_COST_RATE = 0.0

# 打印开关
DEFAULT_VERBOSE_STOCK = False   # 每只股票会非常多，建议 False
DEFAULT_VERBOSE_DAY = True      # 每日净值/盈亏
DEFAULT_PRINT_TOPK = True       # 每日打印入选TopK概览

# 绘图保存
DEFAULT_SAVE_FIG = True
DEFAULT_FIG_DIR = "output/figs"


def _cfg(name, default):
    try:
        from src import config
        return getattr(config, name, default)
    except Exception:
        return default


FEATURE_COLS = _cfg("FEATURE_COLS", DEFAULT_FEATURES)
BACKTEST_START = _cfg("BACKTEST_START", DEFAULT_START)
BACKTEST_END = _cfg("BACKTEST_END", DEFAULT_END)

MIN_TRAIN_SIZE = int(_cfg("MIN_TRAIN_SIZE", DEFAULT_MIN_TRAIN))
TOP_K = int(_cfg("TOP_K", DEFAULT_TOP_K))

HOLD_N = int(_cfg("HOLD_N", DEFAULT_HOLD_N))  # 你的 N
TARGET_COL = _cfg("PORTFOLIO_TARGET_COL", DEFAULT_TARGET_COL)
REALIZED_RET_COL = _cfg("REALIZED_RET_COL", DEFAULT_REALIZED_RET_COL)

WEIGHT_MODE = _cfg("WEIGHT_MODE", DEFAULT_WEIGHT_MODE)
SOFTMAX_TAU = float(_cfg("SOFTMAX_TAU", DEFAULT_SOFTMAX_TAU))

COST_RATE = float(_cfg("COST_RATE", DEFAULT_COST_RATE))

VERBOSE_STOCK = bool(_cfg("VERBOSE_STOCK", DEFAULT_VERBOSE_STOCK))
VERBOSE_DAY = bool(_cfg("VERBOSE_DAY", DEFAULT_VERBOSE_DAY))
PRINT_TOPK = bool(_cfg("PRINT_TOPK", DEFAULT_PRINT_TOPK))

SAVE_FIG = bool(_cfg("SAVE_FIG", DEFAULT_SAVE_FIG))
FIG_DIR = _cfg("FIG_DIR", DEFAULT_FIG_DIR)


# -----------------------------
# 模型训练（回归）安全封装
# -----------------------------
def _normalize_model_type(mt: str) -> str:
    mt = (mt or "").lower().strip()
    if mt in ["xgb_reg", "xgboost_reg", "xgbreg", "xgb"]:
        return "xgboost"
    if mt in ["rf_reg", "randomforest_reg", "rfreg", "rf"]:
        return "randomforest"
    if mt in ["ridge"]:
        return "ridge"
    if mt in ["randomforest"]:
        return "randomforest"
    if mt in ["xgboost"]:
        return "xgboost"
    return mt


def _train_reg_model_safe(X_train, y_train, model_type_norm):
    from src import model as model_mod

    if model_type_norm == "randomforest":
        if hasattr(model_mod, "train_rf_reg"):
            return model_mod.train_rf_reg(X_train, y_train)
        raise ValueError("model.py 未实现 train_rf_reg(X,y)")

    if model_type_norm == "xgboost":
        if hasattr(model_mod, "train_xgb_reg"):
            return model_mod.train_xgb_reg(X_train, y_train)
        raise ValueError("model.py 未实现 train_xgb_reg(X,y)")

    if model_type_norm == "ridge":
        if hasattr(model_mod, "train_ridge_reg"):
            return model_mod.train_ridge_reg(X_train, y_train)
        raise ValueError("model.py 未实现 train_ridge_reg(X,y)")

    raise ValueError("MODEL_TYPE_REG 必须是 rf_reg/xgb_reg/ridge（或其别名）")


# -----------------------------
# 权重方案
# -----------------------------
def _calc_weights(pred_values: np.ndarray) -> np.ndarray:
    """
    输入：TopK 的预测收益（可以为负）
    输出：TopK 权重，和为1
    """
    k = len(pred_values)
    if k == 0:
        return np.array([])

    if WEIGHT_MODE == "equal":
        return np.ones(k) / k

    if WEIGHT_MODE == "proportional":
        x = np.clip(pred_values, 0.0, None)
        if x.sum() <= 1e-12:
            return np.ones(k) / k
        return x / x.sum()

    if WEIGHT_MODE == "softmax":
        tau = max(1e-6, SOFTMAX_TAU)
        z = pred_values / tau
        z = z - np.max(z)
        w = np.exp(z)
        if w.sum() <= 1e-12:
            return np.ones(k) / k
        return w / w.sum()

    return np.ones(k) / k


# -----------------------------
# 分桶结构
# -----------------------------
class Bucket:
    def __init__(self, value: float):
        self.value = float(value)
        self.active = False
        self.symbols = []
        self.weights = None
        self.remaining = 0  # 还要再持有多少个“1日收益步”

    def open(self, symbols, weights, hold_steps: int, cost_rate: float):
        self.active = True
        self.symbols = list(symbols)
        self.weights = np.array(weights, dtype=float)
        self.remaining = int(hold_steps)

        # 简化：开仓扣一次成本（单边）
        if cost_rate > 0:
            self.value *= (1.0 - cost_rate)

    def step_return(self, all_stock_dfs, date, realized_ret_col: str):
        """
        用当日真实收益（date -> date+1）更新桶价值
        注意：ret_1d_fwd 是 date 当天收盘到下一天收盘的收益
        """
        if not self.active:
            return 0.0

        rets = []
        wts = []
        for s, w in zip(self.symbols, self.weights):
            df = all_stock_dfs.get(s)
            if df is None or date not in df.index:
                continue
            r = df.loc[date, realized_ret_col]
            if pd.isna(r):
                continue
            rets.append(float(r))
            wts.append(float(w))

        if len(rets) == 0:
            r_port = 0.0
        else:
            wts = np.array(wts)
            wts = wts / wts.sum()
            r_port = float(np.dot(wts, np.array(rets)))

        before = self.value
        self.value *= (1.0 + r_port)

        self.remaining -= 1
        if self.remaining <= 0:
            self.active = False
            self.symbols = []
            self.weights = None
            self.remaining = 0

        return self.value - before


# -----------------------------
# 主回测：分桶滚动（每天交易）
# -----------------------------
def backtest_topk_portfolio_rollover(initial_cash=1_000_000.0):
    """
    分桶滚动 Top-K 回测（每天交易，多桶持有）
    - Day t 收盘后：预测未来 N 天(TARGET_COL)，选 TopK -> 生成 pending_pick
    - Day t+1 收盘：用一个空闲桶开仓（买入 Day t 的 TopK），并开始从 t+1 -> t+2 获得真实收益
    - 每天所有 active 桶用 REALIZED_RET_COL 做真实记账
    """

    from src import visualization as vz  # 结束时画图

    model_type_norm = _normalize_model_type(MODEL_TYPE_REG)
    hold_steps = max(1, HOLD_N - 1)  # 分桶数 = N-1

    print("\n==============================")
    print("🚀 开始 分桶滚动 Top-K 回测（每天交易，多桶持有）")
    print(f"指数: {INDEX_CODE}")
    print(f"模型: {MODEL_TYPE_REG} (norm -> {model_type_norm})")
    print(f"回测区间: {BACKTEST_START} ~ {BACKTEST_END}")
    print(f"TopK: {TOP_K}")
    print(f"N(预测/持有窗口): {HOLD_N}  -> 分桶数 = N-1 = {hold_steps}")
    print(f"预测目标(用于选股排序): {TARGET_COL}")
    print(f"真实日收益(用于记账): {REALIZED_RET_COL}")
    print(f"权重方式: {WEIGHT_MODE}")
    print(f"开仓成本(单边简化): {COST_RATE:.4%}")
    print("==============================\n")

    # 预测记录（用于散点图）
    pred_records = []

    # 1) 股票池
    symbols = get_index_constituents(INDEX_CODE)
    print(f"📌 成分股数量: {len(symbols)}")

    # 2) 预加载数据
    all_stock_dfs = {}
    for i, s in enumerate(symbols, 1):
        df = get_stock_history(s)
        df = add_features(df)
        all_stock_dfs[s] = df
        if i % 10 == 0 or i == len(symbols):
            print(f"  已加载 {i}/{len(symbols)} 只股票...")

    # 3) 交易日序列（用第一只股票基准）
    base_df = all_stock_dfs[symbols[0]].loc[BACKTEST_START:BACKTEST_END]
    dates = base_df.index
    if len(dates) < 3:
        raise RuntimeError("回测区间交易日太少")

    # 最后一天无法用 ret_1d_fwd 记账（没有 t+1）
    dates_for_steps = dates[:-1]

    # 4) 初始化桶（N-1 份资金）
    bucket_cash = float(initial_cash) / hold_steps
    buckets = [Bucket(bucket_cash) for _ in range(hold_steps)]

    # 5) Day t 信号 -> Day t+1 执行
    pending_pick = None  # (symbols, weights, top_df)

    details = []
    t0 = time.time()
    pbar = tqdm(dates_for_steps, desc="📅 回测进度(按日期)", total=len(dates_for_steps))

    prev_total = float(initial_cash)

    for di, date in enumerate(pbar, 1):
        day_start = time.time()

        # (A) 先：用当日真实收益更新所有 active 桶（date -> date+1）
        pnl_from_hold = 0.0
        for b in buckets:
            pnl_from_hold += b.step_return(all_stock_dfs, date, REALIZED_RET_COL)

        # (B) 再：执行“昨日信号”的开仓（date 收盘开仓，从下一天收益开始体现）
        did_open = False
        opened_bucket_idx = None
        if pending_pick is not None:
            pick_symbols, pick_weights, pick_topdf = pending_pick

            free_idx = None
            for i, b in enumerate(buckets):
                if not b.active:
                    free_idx = i
                    break

            if free_idx is not None:
                buckets[free_idx].open(
                    symbols=pick_symbols,
                    weights=pick_weights,
                    hold_steps=hold_steps,
                    cost_rate=COST_RATE
                )
                did_open = True
                opened_bucket_idx = free_idx

            pending_pick = None

        # (C) 当天收盘后：训练+预测，生成“明天要开的仓”
        preds = []
        for si, (s, df) in enumerate(all_stock_dfs.items(), 1):
            if date not in df.index:
                continue
            try:
                train_df = df.loc[:date].iloc[:-1]
                if len(train_df) < MIN_TRAIN_SIZE:
                    continue

                if TARGET_COL not in df.columns:
                    raise ValueError(f"缺少预测目标列 {TARGET_COL}")
                if REALIZED_RET_COL not in df.columns:
                    raise ValueError(f"缺少真实收益列 {REALIZED_RET_COL}（请在 add_features 里加 ret_1d_fwd）")

                X_train = train_df[FEATURE_COLS]
                y_train = train_df[TARGET_COL]
                model = _train_reg_model_safe(X_train, y_train, model_type_norm)

                X_test = df.loc[[date], FEATURE_COLS]
                pred = float(model.predict(X_test)[0])

                # 仅用于调试：date 的标签真值（不要用于记账）
                true_target = float(df.loc[date, TARGET_COL])

                preds.append((s, pred, true_target))

                # 记录散点图数据
                pred_records.append({
                    "date": pd.to_datetime(date),
                    "symbol": s,
                    "pred": pred,
                    "true_ret": true_target
                })

                if VERBOSE_STOCK:
                    print(f"   [{date.date()}] {s} pred={pred:+.4%} true({TARGET_COL})={true_target:+.4%}")

            except Exception as e:
                if VERBOSE_STOCK:
                    print(f"⚠️ {s} 训练失败: {repr(e)}")
                continue

        opened_k = 0
        top_preview = None

        if len(preds) >= TOP_K:
            pred_df = pd.DataFrame(preds, columns=["symbol", "pred", "true_target"])
            top_df = pred_df.sort_values("pred", ascending=False).head(TOP_K).reset_index(drop=True)

            w = _calc_weights(top_df["pred"].values)
            pending_pick = (top_df["symbol"].tolist(), w, top_df)

            opened_k = TOP_K
            top_preview = top_df
        else:
            pending_pick = None

        # (D) 计算总资金（所有桶价值之和）
        total_balance = float(sum(b.value for b in buckets))
        pnl = total_balance - prev_total
        pct = (pnl / prev_total) if prev_total != 0 else 0.0

        invested_ratio = float(sum(1.0 for b in buckets if b.active) / len(buckets))

        # 计时 & 进度条
        day_cost = time.time() - day_start
        elapsed = time.time() - t0
        avg_per_day = elapsed / di
        remaining = avg_per_day * (len(dates_for_steps) - di)

        pbar.set_postfix({
            "day_s": f"{day_cost:.1f}",
            "elapsed_m": f"{elapsed/60:.1f}",
            "eta_m": f"{remaining/60:.1f}",
            "open": "Y" if did_open else "N",
            "topk": opened_k,
            "active": f"{sum(1 for b in buckets if b.active)}"
        })

        if VERBOSE_DAY:
            print(
                f"\n✅ {date.date()} | 总资金={total_balance:,.2f} | "
                f"日盈亏={pnl:+,.2f} ({pct:+.4%}) | "
                f"持有桶={sum(1 for b in buckets if b.active)}/{len(buckets)} | "
                f"当日耗时={day_cost:.1f}s | 累计={elapsed/60:.1f}min | 预计剩余={remaining/60:.1f}min"
            )

            if PRINT_TOPK and top_preview is not None:
                show_n = min(5, TOP_K)
                print("   明日待开仓 TopK（前5预览）:")
                for i in range(show_n):
                    row = top_preview.iloc[i]
                    print(f"    - {row['symbol']} | pred={row['pred']:+.4%} | true({TARGET_COL})={row['true_target']:+.4%}")

        details.append({
            "date": pd.to_datetime(date),
            "total_balance": total_balance,
            "pnl": pnl,
            "daily_pct": pct,
            "equity": total_balance / float(initial_cash),  # 归一化净值
            "opened_bucket": int(opened_bucket_idx) if did_open else -1,
            "active_buckets": int(sum(1 for b in buckets if b.active)),
            "invested_ratio": invested_ratio,
            "topk_generated": int(opened_k),
            "pnl_from_hold": float(pnl_from_hold),
        })

        prev_total = total_balance

    # 回测结束：统计
    details_df = pd.DataFrame(details).sort_values("date").reset_index(drop=True)

    final_balance = float(details_df["total_balance"].iloc[-1])
    total_pnl = final_balance - float(initial_cash)
    total_return = total_pnl / float(initial_cash)

    # 年化（基于有效交易日）
    ann = float((final_balance / float(initial_cash)) ** (252 / len(details_df)) - 1.0)

    # 最大回撤（基于归一化净值 equity）
    eq = details_df["equity"]
    dd = (eq / eq.cummax() - 1.0)
    max_dd = float(dd.min())

    pred_records_df = pd.DataFrame(pred_records)

    # =========================
    # 自动绘图（保存到 output/figs）
    # =========================
    def fig(name: str):
        return f"{FIG_DIR}/{name}.png" if SAVE_FIG else None

    try:
        vz.plot_equity_curve(details_df,
                             benchmark_df=None,
                             title="策略净值曲线（Strategy）",
                             save_path=fig("01_equity_curve"))

        vz.plot_drawdown_curve(details_df,
                               title="回撤曲线（Drawdown）",
                               save_path=fig("02_drawdown"))

        vz.plot_return_hist(details_df,
                            col="daily_pct",
                            bins=60,
                            title="策略日收益分布直方图（daily_pct）",
                            save_path=fig("03_return_hist"))

        vz.plot_total_balance(details_df,
                              title="每日总余额（Total Balance）",
                              save_path=fig("04_total_balance"))

        vz.plot_daily_pnl(details_df,
                          title="每日盈亏（PnL = ΔBalance）",
                          save_path=fig("05_daily_pnl"))

        vz.plot_pred_vs_true_scatter(pred_records_df,
                                     title=f"预测收益 vs 实际收益（{TARGET_COL}）",
                                     save_path=fig("06_pred_vs_true_scatter"))

        vz.plot_cash_utilization(details_df,
                                 title="资金利用率（Invested Ratio）",
                                 save_path=fig("07_invested_ratio"))

        # turnover 你目前没有严格定义（需要交易前后仓位变化），这里先不画
        # vz.plot_turnover(details_df, title="换手率（Turnover）", save_path=fig("08_turnover"))

        if SAVE_FIG:
            print(f"\n📌 图表已保存到：{FIG_DIR}/")
    except Exception as e:
        print(f"⚠️ 绘图失败（不影响回测结果）：{repr(e)}")

    result = {
        "equity": details_df["equity"],
        "annual_return": ann,
        "max_drawdown": max_dd,
        "final_equity": final_balance,     # 为兼容你之前字段名，这里保留
        "final_balance": final_balance,
        "total_pnl": total_pnl,
        "total_return": total_return,
        "details": details_df,
        "pred_records": pred_records_df
    }
    return result
