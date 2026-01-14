import ta
import pandas as pd


def add_features(df: pd.DataFrame):
    """添加技术指标特征"""

    df = df.copy()

    # 均线
    df["MA5"] = df["收盘"].rolling(5).mean()
    df["MA10"] = df["收盘"].rolling(10).mean()
    df["MA20"] = df["收盘"].rolling(20).mean()

    # MACD
    df["DIF"] = ta.trend.ema_indicator(df["收盘"], 12) - ta.trend.ema_indicator(df["收盘"], 26)
    df["DEA"] = ta.trend.ema_indicator(df["DIF"], 9)
    df["MACD"] = 2 * (df["DIF"] - df["DEA"])

    # 成交量
    df["VOL_MA5"] = df["成交量"].rolling(5).mean()

    # 收益率与波动率
    df["Return"] = df["收盘"].pct_change()
    df["Volatility"] = df["Return"].rolling(10).std()

    # 标签：下一天是否上涨
    df["Target"] = (df["收盘"].shift(-1) > df["收盘"]).astype(int)

    # 未来 1 / 5 / 10 日收益
    df["ret_1d_fwd"] = df["收盘"].shift(-1) / df["收盘"] - 1
    df["ret_1d_open_fwd"] = df["开盘"].shift(-1) / df["开盘"] - 1
    df["ret_5d"] = df["收盘"].shift(-5) / df["收盘"] - 1
    df["ret_10d"] = df["收盘"].shift(-10) / df["收盘"] - 1

    return df


def add_index_features(stock_df, index_df):
    df = stock_df.join(index_df["收盘"].rename("HS300_Close"), how="left")

    df["HS300_MA5"] = df["HS300_Close"].rolling(5).mean()
    df["HS300_MA20"] = df["HS300_Close"].rolling(20).mean()
    df["HS300_Return"] = df["HS300_Close"].pct_change()
    df["HS300_Volatility"] = df["HS300_Return"].rolling(10).std()

    return df
