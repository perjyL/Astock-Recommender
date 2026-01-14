# src/backtest_portfolio.py
import os
import time
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.data_loader import get_index_constituents, get_stock_history
from src.feature_engineering import add_features
from src.config import INDEX_CODE, MODEL_TYPE_REG

# -----------------------------
# 默认参数
# -----------------------------
DEFAULT_FEATURES = ["MA5", "MA10", "MA20", "MACD", "DIF", "DEA", "VOL_MA5", "Volatility"]
DEFAULT_START = "2025-01-01"
DEFAULT_END = "2025-12-31"
DEFAULT_MIN_TRAIN = 200
DEFAULT_TOP_K = 20

DEFAULT_HOLD_N = 5
DEFAULT_TARGET_COL = "ret_5d"

# ✅ 改成：开盘->开盘 的真实收益（用于记账）
DEFAULT_REALIZED_RET_COL = "ret_1d_open_fwd"

DEFAULT_WEIGHT_MODE = "equal"
DEFAULT_SOFTMAX_TAU = 1.0
DEFAULT_COST_RATE = 0.0

DEFAULT_VERBOSE_STOCK = False
DEFAULT_VERBOSE_DAY = True
DEFAULT_PRINT_TOPK = True

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

HOLD_N = int(_cfg("HOLD_N", DEFAULT_HOLD_N))
TARGET_COL = _cfg("PORTFOLIO_TARGET_COL", DEFAULT_TARGET_COL)

# ✅ 这里就是开盘收益列
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
        self.remaining = 0  # 还要持有多少个“开盘->开盘”步

    def open_at_next_open(self, symbols, weights, hold_steps: int, cost_rate: float):
        """在“今日开盘”执行开仓（成交价口径是开盘）"""
        self.active = True
        self.symbols = list(symbols)
        self.weights = np.array(weights, dtype=float)
        self.remaining = int(hold_steps)

        # 简化：开仓扣一次成本（单边）
        if cost_rate > 0:
            self.value *= (1.0 - cost_rate)

    def settle_open_to_open(self, all_stock_dfs, settle_date, realized_ret_col: str) -> float:
        """
        结算：settle_date 的 开盘 -> settle_date+1 的 开盘
        realized_ret_col = ret_1d_open_fwd（存放在 settle_date 行）
        """
        if not self.active:
            return 0.0

        rets, wts = [], []
        for s, w in zip(self.symbols, self.weights):
            df = all_stock_dfs.get(s)
            if df is None or settle_date not in df.index:
                continue
            r = df.loc[settle_date, realized_ret_col]
            if pd.isna(r):
                continue
            rets.append(float(r))
            wts.append(float(w))

        if len(rets) == 0:
            r_port = 0.0
        else:
            wts = np.array(wts, dtype=float)
            wts = wts / wts.sum()
            r_port = float(np.dot(wts, np.array(rets, dtype=float)))

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
# 主回测：信号收盘生成，次日开盘成交
# -----------------------------
def backtest_topk_portfolio_rollover(initial_cash=1_000_000.0):
    """
    时间线（严格、无未来数据泄露）：

    Day t（收盘后）：
      - 用 <= t 的特征做预测（pred 发生在收盘后）
      - 生成“明天开盘要买”的 TopK 信号（pending_pick）

    Day t+1（开盘时刻）：
      1) 先结算所有持仓桶的 (t 开盘 -> t+1 开盘) 收益：用 ret_1d_open_fwd(t)
      2) 再执行 pending_pick（在 t+1 开盘开仓）
      3) 等到 t+1 收盘，再生成下一天的 pending_pick
    """

    from src import visualization as vz

    model_type_norm = _normalize_model_type(MODEL_TYPE_REG)
    hold_steps = max(1, HOLD_N - 1)

    print("\n==============================")
    print("🚀 分桶滚动 Top-K 回测（信号=收盘，成交=次日开盘）")
    print(f"指数: {INDEX_CODE}")
    print(f"模型: {MODEL_TYPE_REG} (norm -> {model_type_norm})")
    print(f"回测区间: {BACKTEST_START} ~ {BACKTEST_END}")
    print(f"TopK: {TOP_K}")
    print(f"N(预测/持有窗口): {HOLD_N} -> 分桶数 = N-1 = {hold_steps}")
    print(f"预测目标(用于排序): {TARGET_COL}")
    print(f"真实收益(用于记账): {REALIZED_RET_COL}  (开盘->开盘)")
    print(f"权重方式: {WEIGHT_MODE}")
    print(f"开仓成本(单边简化): {COST_RATE:.4%}")
    print("==============================\n")

    # 预测记录（散点图）
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
        print(df)
        if i % 10 == 0 or i == len(symbols):
            print(f"  已加载 {i}/{len(symbols)} 只股票...")


    # 3) 交易日序列
    base_df = all_stock_dfs[symbols[0]].loc[BACKTEST_START:BACKTEST_END]
    dates = base_df.index
    if len(dates) < 5:
        raise RuntimeError("回测区间交易日太少")

    # 4) 初始化桶（N-1份资金）
    bucket_cash = float(initial_cash) / hold_steps
    buckets = [Bucket(bucket_cash) for _ in range(hold_steps)]

    # pending_pick：由“昨日收盘”产生，在“今日开盘”成交
    pending_pick = None  # (symbols, weights, top_df)

    details = []
    t0 = time.time()
    pbar = tqdm(dates, desc="📅 回测进度(按日期)", total=len(dates))

    prev_total = float(initial_cash)
    prev_trade_date = None  # 用于结算 prev_trade_date 开盘->今日开盘

    for di, date in enumerate(pbar, 1):
        day_start = time.time()

        # =============== (A) 今日开盘：先结算昨日开盘->今日开盘 =================
        pnl_from_hold = 0.0
        if prev_trade_date is not None:
            for b in buckets:
                pnl_from_hold += b.settle_open_to_open(all_stock_dfs, prev_trade_date, REALIZED_RET_COL)

        # =============== (B) 今日开盘：执行昨日收盘信号，成交在今日开盘 =================
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
                buckets[free_idx].open_at_next_open(
                    symbols=pick_symbols,
                    weights=pick_weights,
                    hold_steps=hold_steps,
                    cost_rate=COST_RATE
                )
                did_open = True
                opened_bucket_idx = free_idx

            pending_pick = None  # 清空

        # =============== (C) 今日收盘后：训练+预测，生成“明日开盘要买”的 pending_pick =================
        preds = []
        for s, df in all_stock_dfs.items():
            if date not in df.index:
                continue
            try:
                # ✅ 训练只用到 date-1（严格不使用未来标签）
                train_df = df.loc[:date].iloc[:-1]
                if len(train_df) < MIN_TRAIN_SIZE:
                    continue

                if TARGET_COL not in df.columns:
                    raise ValueError(f"缺少预测目标列 {TARGET_COL}")
                if REALIZED_RET_COL not in df.columns:
                    raise ValueError(f"缺少真实收益列 {REALIZED_RET_COL}（请在 add_features 里生成）")

                X_train = train_df[FEATURE_COLS]
                y_train = train_df[TARGET_COL]
                model = _train_reg_model_safe(X_train, y_train, model_type_norm)

                # ✅ 用 date 当天特征做预测：信号在“date 收盘后”生成
                X_test = df.loc[[date], FEATURE_COLS]
                pred = float(model.predict(X_test)[0])
                true_target = float(df.loc[date, TARGET_COL])  # 仅用于散点图/诊断

                preds.append((s, pred, true_target))
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

        # =============== (D) 今日收盘：输出“截至今日开盘结算后的资产”口径 =================
        # 这里的 total_balance 是“今日开盘已结算 + 今日开盘已成交后”的余额（不含今日开盘->明日开盘收益）
        total_balance = float(sum(b.value for b in buckets))
        pnl = total_balance - prev_total
        pct = pnl / prev_total if prev_total != 0 else 0.0
        invested_ratio = float(sum(1.0 for b in buckets if b.active) / len(buckets))

        day_cost = time.time() - day_start
        elapsed = time.time() - t0
        avg_per_day = elapsed / di
        remaining = avg_per_day * (len(dates) - di)

        pbar.set_postfix({
            "day_s": f"{day_cost:.1f}",
            "elapsed_m": f"{elapsed/60:.1f}",
            "eta_m": f"{remaining/60:.1f}",
            "open": "Y" if did_open else "N",
            "topk": opened_k,
            "active": f"{sum(1 for b in buckets if b.active)}"
        })

        if VERBOSE_DAY:
            settle_info = f"{prev_trade_date.date()}开→{date.date()}开" if prev_trade_date is not None else "N/A(首日无结算)"
            print(
                f"\n✅ {date.date()} | 结算={settle_info} | 总资金={total_balance:,.2f} | "
                f"日盈亏={pnl:+,.2f} ({pct:+.4%}) | "
                f"持有桶={sum(1 for b in buckets if b.active)}/{len(buckets)} | "
                f"当日耗时={day_cost:.1f}s | 累计={elapsed/60:.1f}min | 预计剩余={remaining/60:.1f}min"
            )

            if PRINT_TOPK and top_preview is not None:
                show_n = min(5, TOP_K)
                print("   明日开盘待开仓 TopK（前5预览）:")
                for i in range(show_n):
                    row = top_preview.iloc[i]
                    print(f"    - {row['symbol']} | pred={row['pred']:+.4%} | true({TARGET_COL})={row['true_target']:+.4%}")

        details.append({
            "date": pd.to_datetime(date),
            "total_balance": total_balance,
            "pnl": pnl,
            "daily_pct": pct,
            "equity": total_balance / float(initial_cash),
            "opened_bucket": int(opened_bucket_idx) if did_open else -1,
            "active_buckets": int(sum(1 for b in buckets if b.active)),
            "invested_ratio": invested_ratio,
            "topk_generated": int(opened_k),
            "pnl_from_hold": float(pnl_from_hold),
            "settle_from": pd.to_datetime(prev_trade_date) if prev_trade_date is not None else pd.NaT
        })

        prev_total = total_balance
        prev_trade_date = date  # 下一天结算用它（date开→next_date开）

    # -----------------------------
    # 回测结束统计
    # -----------------------------
    details_df = pd.DataFrame(details).sort_values("date").reset_index(drop=True)
    final_balance = float(details_df["total_balance"].iloc[-1])
    total_pnl = final_balance - float(initial_cash)
    total_return = total_pnl / float(initial_cash)
    ann = float((final_balance / float(initial_cash)) ** (252 / len(details_df)) - 1.0)

    eq = details_df["equity"]
    dd = (eq / eq.cummax() - 1.0)
    max_dd = float(dd.min())

    pred_records_df = pd.DataFrame(pred_records)

    # -----------------------------
    # 画图
    # -----------------------------
    def _ensure_dir(p: str):
        if p and not os.path.exists(p):
            os.makedirs(p, exist_ok=True)

    def fig(name: str):
        if not SAVE_FIG:
            return None
        _ensure_dir(FIG_DIR)
        return f"{FIG_DIR}/{name}.png"

    try:
        vz.plot_equity_curve(details_df, benchmark_df=None,
                             title="策略净值曲线（开盘成交口径）", save_path=fig("01_equity_curve"))
        vz.plot_drawdown_curve(details_df, title="回撤曲线（Drawdown）", save_path=fig("02_drawdown"))
        vz.plot_return_hist(details_df, col="daily_pct", bins=60,
                            title="策略日收益分布（daily_pct）", save_path=fig("03_return_hist"))
        vz.plot_total_balance(details_df, title="每日总余额（Total Balance）", save_path=fig("04_total_balance"))
        vz.plot_daily_pnl(details_df, title="每日盈亏（PnL）", save_path=fig("05_daily_pnl"))

        if len(pred_records_df) > 0:
            vz.plot_pred_vs_true_scatter(pred_records_df,
                                         title=f"预测收益 vs 实际收益（{TARGET_COL}）",
                                         save_path=fig("06_pred_vs_true_scatter"))

        vz.plot_cash_utilization(details_df, title="资金利用率（Invested Ratio）", save_path=fig("07_invested_ratio"))

        if SAVE_FIG:
            print(f"\n📌 图表已保存到：{FIG_DIR}/")
    except Exception as e:
        print(f"⚠️ 绘图失败（不影响回测结果）：{repr(e)}")

    return {
        "equity": details_df["equity"],
        "annual_return": ann,
        "max_drawdown": max_dd,
        "final_equity": final_balance,
        "final_balance": final_balance,
        "total_pnl": total_pnl,
        "total_return": total_return,
        "details": details_df,
        "pred_records": pred_records_df
    }
