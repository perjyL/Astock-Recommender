# src/backtest_portfolio.py
import time
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.data_loader import get_index_constituents, get_stock_history
from src.feature_engineering import add_features
from src.config import INDEX_CODE, MODEL_TYPE_REG

DEFAULT_FEATURES = [
    "MA5", "MA10", "MA20",
    "MACD", "DIF", "DEA",
    "VOL_MA5", "Volatility"
]

# 预测目标（用于排序选股）
DEFAULT_TARGET_COL = "ret_5d"
# 真实日收益（用于每天记账）
DEFAULT_REALIZED_RET_COL = "ret_1d_fwd"

DEFAULT_TOP_K = 20
DEFAULT_MIN_TRAIN = 200
DEFAULT_START = "2025-01-01"
DEFAULT_END = "2025-12-31"

# 调仓频率：1=每日调仓；5=5日调仓
DEFAULT_REBALANCE_N = 1

# 交易成本/滑点（单边），例如 0.001=千分之一
DEFAULT_COST_RATE = 0.0


def _cfg(name, default):
    try:
        from src import config
        return getattr(config, name, default)
    except Exception:
        return default


FEATURE_COLS = _cfg("FEATURE_COLS", DEFAULT_FEATURES)

TARGET_COL = _cfg("PORTFOLIO_TARGET_COL", DEFAULT_TARGET_COL)          # 用于预测&排序
REALIZED_RET_COL = _cfg("REALIZED_RET_COL", DEFAULT_REALIZED_RET_COL)  # 用于真实记账

TOP_K = int(_cfg("TOP_K", DEFAULT_TOP_K))
MIN_TRAIN_SIZE = int(_cfg("MIN_TRAIN_SIZE", DEFAULT_MIN_TRAIN))
BACKTEST_START = _cfg("BACKTEST_START", DEFAULT_START)
BACKTEST_END = _cfg("BACKTEST_END", DEFAULT_END)

REBALANCE_N = int(_cfg("REBALANCE_N", DEFAULT_REBALANCE_N))
COST_RATE = float(_cfg("COST_RATE", DEFAULT_COST_RATE))

# 是否打印每只股票
VERBOSE_STOCK = bool(_cfg("VERBOSE_STOCK", True))
# 是否打印每日汇总
VERBOSE_DATE = bool(_cfg("VERBOSE_DATE", True))


def _normalize_model_type(mt: str) -> str:
    """把各种别名统一到我们内部的回归模型名"""
    mt = (mt or "").lower().strip()

    # 回归别名 -> 内部统一名
    if mt in ["xgb_reg", "xgboost_reg", "xgbreg", "xgb"]:
        return "xgboost"
    if mt in ["rf_reg", "randomforest_reg", "rfreg", "rf"]:
        return "randomforest"

    # 兼容
    if mt in ["randomforest"]:
        return "randomforest"
    if mt in ["xgboost"]:
        return "xgboost"

    return mt


def _train_reg_model_safe(X_train, y_train, model_type_norm):
    """
    组合回测预测未来收益(ret_5d/ret_10d)，应训练回归模型：
      - train_rf_reg(X,y)
      - train_xgb_reg(X,y)
      - train_ridge_reg(X,y) 也可扩展
    """
    from src import model as model_mod

    if model_type_norm == "randomforest":
        if hasattr(model_mod, "train_rf_reg"):
            return model_mod.train_rf_reg(X_train, y_train)
        raise ValueError("你的 model.py 未实现 train_rf_reg(X,y)，无法用于组合回归回测")

    if model_type_norm == "xgboost":
        if hasattr(model_mod, "train_xgb_reg"):
            return model_mod.train_xgb_reg(X_train, y_train)
        raise ValueError("你的 model.py 未实现 train_xgb_reg(X,y)，无法用于组合回归回测")

    if model_type_norm == "ridge":
        if hasattr(model_mod, "train_ridge_reg"):
            return model_mod.train_ridge_reg(X_train, y_train)
        raise ValueError("你的 model.py 未实现 train_ridge_reg(X,y)，无法用于组合回归回测")

    raise ValueError("MODEL_TYPE_REG 必须是 randomforest/xgboost/ridge（或其 *_reg 别名）才能用于组合回归回测")


def backtest_topk_portfolio():
    """
    Top-K 多头组合回测：
    - 预测列（用于选股排序）：TARGET_COL（ret_5d/ret_10d）
    - 真实记账（日收益）：REALIZED_RET_COL（ret_1d_fwd）
    - 调仓频率：REBALANCE_N（1=每日调仓；5=5日调仓）
    - 成本：COST_RATE（在调仓日按换手扣减，简化版）
    """
    model_type_norm = _normalize_model_type(MODEL_TYPE_REG)

    print("\n==============================")
    print("🚀 开始 Top-K 组合回测（多头）")
    print(f"指数: {INDEX_CODE}")
    print(f"模型: {MODEL_TYPE_REG}  (norm -> {model_type_norm})")
    print(f"回测区间: {BACKTEST_START} ~ {BACKTEST_END}")
    print(f"TopK: {TOP_K}")
    print(f"预测目标(用于选股): {TARGET_COL}")
    print(f"真实日收益(用于记账): {REALIZED_RET_COL}")
    print(f"调仓频率: 每 {REBALANCE_N} 个交易日调仓一次")
    print(f"最小训练样本: {MIN_TRAIN_SIZE}")
    print(f"单边成本/滑点(简化): {COST_RATE:.4%}")
    print("==============================\n")

    symbols = get_index_constituents(INDEX_CODE)
    print(f"📌 成分股数量: {len(symbols)}")

    # 预加载数据
    all_stock_dfs = {}
    for i, s in enumerate(symbols, 1):
        df = get_stock_history(s)
        df = add_features(df)
        all_stock_dfs[s] = df
        if i % 10 == 0 or i == len(symbols):
            print(f"  已加载 {i}/{len(symbols)} 只股票...")

    # 取交易日序列（用第一只股票作为基准）
    base_df = all_stock_dfs[symbols[0]].loc[BACKTEST_START:BACKTEST_END]
    dates = base_df.index
    if len(dates) == 0:
        raise RuntimeError("回测区间内无交易日数据，请检查 START/END 或数据源")

    # 回测输出
    portfolio_returns = []
    used_dates = []
    daily_details = []

    # 持仓缓存（用于 N 日调仓）
    holding_symbols = None
    holding_pred_df = None  # 记录调仓日的预测/真实信息（可选）

    t0 = time.time()
    pbar = tqdm(dates, desc="📅 回测进度(按日期)", total=len(dates))

    for di, date in enumerate(pbar, 1):
        day_start = time.time()

        # 是否为调仓日
        is_rebalance = ((di - 1) % REBALANCE_N == 0) or (holding_symbols is None)

        # 1) 调仓日：训练+预测+选股（用 TARGET_COL 预测排序）
        if is_rebalance:
            preds = []

            for si, (s, df) in enumerate(all_stock_dfs.items(), 1):
                if date not in df.index:
                    continue

                try:
                    # 训练区间：到 date 的前一日（避免未来信息）
                    train_df = df.loc[:date].iloc[:-1]
                    if len(train_df) < MIN_TRAIN_SIZE:
                        continue

                    if TARGET_COL not in df.columns:
                        raise ValueError(f"缺少预测目标列 {TARGET_COL}，请确认 add_features 已生成")
                    if REALIZED_RET_COL not in df.columns:
                        raise ValueError(f"缺少真实收益列 {REALIZED_RET_COL}，请在 add_features 加入 ret_1d_fwd")

                    X_train = train_df[FEATURE_COLS]
                    y_train = train_df[TARGET_COL]

                    model = _train_reg_model_safe(X_train, y_train, model_type_norm)

                    X_test = df.loc[[date], FEATURE_COLS]
                    pred_ret = float(model.predict(X_test)[0])

                    # 这里不用于记账，仅用于调试显示：当日标签列（ret_5d）真实值
                    true_target = float(df.loc[date, TARGET_COL])

                    preds.append((s, pred_ret, true_target))

                    if VERBOSE_STOCK:
                        print(f"   [REB {date.date()}] ({si:03d}/{len(all_stock_dfs)}) {s} | "
                              f"pred({TARGET_COL})={pred_ret:+.4%} | true({TARGET_COL})={true_target:+.4%}")

                except Exception as e:
                    if VERBOSE_STOCK:
                        print(f"⚠️ {s} 训练失败: {repr(e)}")
                    continue

            if len(preds) < TOP_K:
                if VERBOSE_DATE:
                    print(f"⚠️ {date.date()} 调仓失败：有效股票不足 top_k：{len(preds)}/{TOP_K}，跳过该日")
                continue

            pred_df = pd.DataFrame(preds, columns=["symbol", "pred", "true_target"])
            top_df = pred_df.sort_values("pred", ascending=False).head(TOP_K)

            new_holding = top_df["symbol"].tolist()

            # 简化成本：在调仓日按“换手率”扣一次（近似）
            # turnover = 1 - overlap_ratio
            if holding_symbols is None:
                turnover = 1.0
            else:
                overlap = len(set(holding_symbols).intersection(set(new_holding)))
                turnover = 1.0 - overlap / max(1, TOP_K)

            holding_symbols = new_holding
            holding_pred_df = top_df.copy()

        else:
            # 非调仓日不重新训练选股
            turnover = 0.0

        # 2) 每个交易日：用持仓的“真实 1 日收益”记账（REALIZED_RET_COL）
        realized_rets = []
        missing = 0

        for s in holding_symbols:
            df = all_stock_dfs.get(s)
            if df is None or date not in df.index:
                missing += 1
                continue

            r = df.loc[date, REALIZED_RET_COL]
            if pd.isna(r):
                # 通常是最后几天因为 shift(-1) 没有下一天价格
                missing += 1
                continue
            realized_rets.append(float(r))

        if len(realized_rets) < max(1, int(0.5 * TOP_K)):
            # 太少则跳过（避免最后一天/缺数据把收益算歪）
            if VERBOSE_DATE:
                print(f"⚠️ {date.date()} 可用于记账的持仓收益不足：{len(realized_rets)}/{TOP_K}，跳过该日")
            continue

        gross_ret = float(np.mean(realized_rets))

        # 成本扣减（简化）：当日净收益 = gross_ret - turnover * COST_RATE * 2
        # *2 近似双边（卖出+买入）；你也可以只用单边
        net_ret = gross_ret - turnover * COST_RATE * 2

        portfolio_returns.append(net_ret)
        used_dates.append(date)

        # 计时统计
        day_cost = time.time() - day_start
        elapsed = time.time() - t0
        avg_per_day = elapsed / di
        remaining = avg_per_day * (len(dates) - di)

        pbar.set_postfix({
            "day_s": f"{day_cost:.1f}",
            "elapsed_m": f"{elapsed/60:.1f}",
            "eta_m": f"{remaining/60:.1f}",
            "reb": "Y" if is_rebalance else "N",
            "turn": f"{turnover:.2f}",
            "use": f"{len(realized_rets)}"
        })

        if VERBOSE_DATE:
            print(
                f"✅ {date.date()} "
                f"{'(调仓)' if is_rebalance else '(持仓)'} "
                f"日收益(gross)={gross_ret:+.4%} | 日收益(net)={net_ret:+.4%} | "
                f"turnover={turnover:.2f} | "
                f"当日耗时={day_cost:.1f}s | 累计={elapsed/60:.1f}min | 预计剩余={remaining/60:.1f}min"
            )

        daily_details.append({
            "date": pd.to_datetime(date),
            "rebalance": bool(is_rebalance),
            "turnover": float(turnover),
            "gross_ret": float(gross_ret),
            "portfolio_ret": float(net_ret),
            "holding_n": int(len(holding_symbols) if holding_symbols else 0),
            "used_ret_n": int(len(realized_rets)),
            "missing_n": int(missing),
        })

    if len(portfolio_returns) == 0:
        raise RuntimeError("portfolio_returns 为空：可能 min_train 太大 / 数据缺失 / top_k 过大 / 最后区间无 ret_1d_fwd")

    details_df = pd.DataFrame(daily_details).sort_values("date")
    details_df["equity"] = (1 + details_df["portfolio_ret"]).cumprod()

    equity = details_df["equity"]
    max_dd = float((equity / equity.cummax() - 1).min())

    # 年化收益：用实际有收益的交易日数量估算
    annual_return = float(equity.iloc[-1] ** (252 / len(equity)) - 1)

    result = {
        "equity": equity,
        "annual_return": annual_return,
        "max_drawdown": max_dd,
        "details": details_df
    }
    return result
