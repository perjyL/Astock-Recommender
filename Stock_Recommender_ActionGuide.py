import argparse
from datetime import datetime
import os
import numpy as np
import pandas as pd

from src.config import (
    INDEX_CODE,
    MODEL_TYPE_REG,
    FEATURE_COLS,
    PORTFOLIO_TARGET_COL,
    HOLD_N,
    TOP_K,
    MIN_TRAIN_SIZE,
    WEIGHT_MODE,
    SOFTMAX_TAU,
    COST_RATE,
)
from src.data_loader import (
    get_index_constituents,
    get_index_constituents_with_name,
    get_hs300_index,
    get_stock_history,
)
from src.feature_engineering import add_features
from src.model import train_model_reg
from src.backtest_portfolio import _calc_weights, _normalize_model_type


def _pick_signal_date(index_code: str):
    """优先用指数行情的最后交易日作为信号日期。"""
    index_df = get_hs300_index(index_code)
    if index_df is not None and not index_df.empty:
        return pd.to_datetime(index_df.index.max())
    return None


def _format_pct(x):
    if x is None or not np.isfinite(x):
        return "N/A"
    return f"{x:.2%}"


def generate_next_day_guide(capital: float = 1_000_000.0, use_realtime: bool = False):
    # 1) 基本信息
    hold_steps = max(1, HOLD_N - 1)
    bucket_ratio = 1.0 / hold_steps
    model_type_norm = _normalize_model_type(MODEL_TYPE_REG)

    print("\n==============================")
    print("📌 明日操作行动指南（基于 backtest_portfolio 逻辑）")
    print(f"指数: {INDEX_CODE}")
    print(f"模型: {MODEL_TYPE_REG} (norm -> {model_type_norm})")
    print(f"TopK: {TOP_K} | 预测目标: {PORTFOLIO_TARGET_COL}")
    print(f"权重方式: {WEIGHT_MODE} | Softmax Tau: {SOFTMAX_TAU}")
    print(f"N(预测/持有窗口): {HOLD_N}  -> 分桶数 = N-1 = {hold_steps}")
    print(f"单边成本假设: {COST_RATE:.2%}")
    print(f"总资金: {capital:,.2f}")
    print("==============================\n")

    # 2) 获取成分股列表
    name_map = get_index_constituents_with_name(INDEX_CODE)
    symbols = list(name_map.keys()) if name_map else get_index_constituents(INDEX_CODE)
    if not symbols:
        print(f"❌ 指数 {INDEX_CODE} 成分股为空，无法生成行动指南")
        return None

    # 3) 信号日期（最后交易日）
    signal_date = _pick_signal_date(INDEX_CODE)
    if signal_date is None:
        print("❌ 未能获取指数交易日，无法生成行动指南")
        return None

    print(f"信号日期（T 日收盘数据）: {signal_date.date()}")
    print("明日执行日期（T+1）: 下一个交易日")

    # 4) 逐股训练与预测
    preds = []
    skipped = 0
    for s in symbols:
        name = name_map.get(s, s)
        try:
            df = get_stock_history(s, use_realtime=use_realtime)
            if df is None or df.empty:
                skipped += 1
                continue

            df = add_features(df)
            if signal_date not in df.index:
                skipped += 1
                continue

            # 使用 signal_date 前一天的数据训练，避免未来信息
            train_df = df.loc[:signal_date].iloc[:-1]
            train_df = train_df[FEATURE_COLS + [PORTFOLIO_TARGET_COL]].dropna()
            if len(train_df) < MIN_TRAIN_SIZE:
                skipped += 1
                continue

            X_train = train_df[FEATURE_COLS]
            y_train = train_df[PORTFOLIO_TARGET_COL]
            model = train_model_reg(X_train, y_train, model_type=MODEL_TYPE_REG)

            X_test = df.loc[[signal_date], FEATURE_COLS].dropna()
            if X_test.empty:
                skipped += 1
                continue

            pred = float(model.predict(X_test)[0])
            if not np.isfinite(pred):
                skipped += 1
                continue

            preds.append((s, name, pred))
        except Exception:
            skipped += 1
            continue

    if len(preds) < TOP_K:
        print(f"❌ 可用样本不足（{len(preds)} < TOP_K={TOP_K}），无法生成行动指南")
        return None

    # 5) 选 TopK + 权重
    pred_df = pd.DataFrame(preds, columns=["Code", "Name", "Pred_Return"])
    top_df = pred_df.sort_values("Pred_Return", ascending=False).head(TOP_K).reset_index(drop=True)
    weights = _calc_weights(top_df["Pred_Return"].values)
    top_df["Weight_in_Bucket"] = weights
    top_df["Capital_in_Bucket"] = capital * bucket_ratio * top_df["Weight_in_Bucket"]
    top_df.insert(0, "Rank", range(1, len(top_df) + 1))

    # 6) 输出行动指南
    print("\n===== 明日行动指南（T+1）=====")
    print("操作原则（与 backtest_portfolio 一致）：")
    print("1) 信号来自 T 日收盘数据，T+1 收盘建仓")
    if hold_steps > 1:
        print(f"2) 资金分 {hold_steps} 份，每日仅投入 {bucket_ratio:.2%} 总资金")
        print(f"3) 每个桶持有 {hold_steps} 个交易日，到期释放资金")
    else:
        print("2) HOLD_N=1：每天可以使用全部资金做单日持有")
    print("4) 权重按策略计算（equal/proportional/softmax）")
    print("5) 以下为明日建议新开仓名单：\n")

    for _, row in top_df.iterrows():
        print(
            f"#{int(row['Rank']):02d} {row['Code']} {row['Name']} | "
            f"预测未来收益={_format_pct(row['Pred_Return'])} | "
            f"桶内权重={row['Weight_in_Bucket']:.2%} | "
            f"建议投入={row['Capital_in_Bucket']:,.2f}"
        )

    # 7) 保存 CSV
    os.makedirs("output", exist_ok=True)
    out_path = "output/next_day_action_guide.csv"
    top_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n✅ 明日行动指南已保存：{out_path}")
    print(f"本次跳过股票数量：{skipped}")
    return top_df


def parse_args():
    parser = argparse.ArgumentParser(description="生成明日操作行动指南（Top-K 策略）")
    parser.add_argument("--capital", type=float, default=1_000_000.0, help="总资金（用于计算建议投入金额）")
    parser.add_argument("--use-realtime", action="store_true", help="使用实时行情覆盖当日数据")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate_next_day_guide(capital=args.capital, use_realtime=args.use_realtime)
