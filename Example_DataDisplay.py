# 数据来源（AkShare）
import akshare as ak
import pandas as pd

# 1.指数成分股（以沪深300为例）
# 获取沪深300成分股
stocks = ak.index_stock_cons_csindex(symbol="000300")

# 2.个股历史行情（日频）
symbol = "603296"
df = ak.stock_zh_a_hist(
    symbol=symbol,
    period="daily",
    start_date="20230101",
    end_date="20260112",
    adjust="qfq"
)


# 金融量化特征

# 1.均线系统（趋势特征）
df["MA5"] = df["收盘"].rolling(5).mean()
df["MA10"] = df["收盘"].rolling(10).mean()
df["MA20"] = df["收盘"].rolling(20).mean()


# 2.MACD（动量指标）
# 金融意义：衡量趋势强弱与拐点
# DIF 上穿 DEA → 看涨信号
import ta

df["DIF"] = ta.trend.ema_indicator(df["收盘"], 12) - ta.trend.ema_indicator(df["收盘"], 26)
df["DEA"] = ta.trend.ema_indicator(df["DIF"], 9)
df["MACD"] = 2 * (df["DIF"] - df["DEA"])

# 3.成交量变化（资金行为）
df["VOL_MA5"] = df["成交量"].rolling(5).mean()

# 4.收益率与波动率（风险刻画）
df["Return"] = df["收盘"].pct_change()
df["Volatility"] = df["Return"].rolling(10).std()

today = pd.Timestamp.today().date()
last_trade_date = pd.to_datetime(df["日期"]).max().date()

if last_trade_date == today:
    try:
        spot = ak.stock_zh_a_spot_em()
        symbol_code = symbol[-6:]
        spot_row = spot.loc[spot["代码"] == symbol_code]
        if not spot_row.empty:
            show_cols = [
                "代码", "名称", "最新价", "涨跌幅", "涨跌额",
                "今开", "最高", "最低", "昨收", "成交量", "成交额"
            ]
            show_cols = [col for col in show_cols if col in spot_row.columns]
            print("\n===== 当日盘中行情 =====")
            print(spot_row[show_cols])
        else:
            print("\n当日盘中行情为空：未匹配到代码")
    except Exception as exc:
        print(f"\n获取当日盘中行情失败：{exc}")

# 设置全局显示选项（显示所有行和列）
pd.set_option('display.max_rows', None)      # 显示所有行
pd.set_option('display.max_columns', None)   # 显示所有列
pd.set_option('display.width', 1000)         # 控制整体宽度
pd.set_option('display.max_colwidth', 100)   # 控制单列最大宽度

print(stocks)
print(df)
