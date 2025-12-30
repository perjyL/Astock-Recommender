import matplotlib.pyplot as plt


def plot_price_ma(df, symbol):
    plt.figure(figsize=(12, 6))
    plt.plot(df["收盘"], label="Close")
    plt.plot(df["MA5"], label="MA5")
    plt.plot(df["MA10"], label="MA10")
    plt.plot(df["MA20"], label="MA20")
    plt.title(f"{symbol} 价格与均线")
    plt.legend()
    plt.show()


def plot_macd(df, symbol):
    plt.figure(figsize=(12, 4))
    plt.bar(df.index, df["MACD"], label="MACD")
    plt.plot(df["DIF"], label="DIF")
    plt.plot(df["DEA"], label="DEA")
    plt.title(f"{symbol} MACD 指标")
    plt.legend()
    plt.show()


def plot_volume(df, symbol):
    plt.figure(figsize=(12, 4))
    plt.bar(df.index, df["成交量"], label="Volume")
    plt.plot(df["VOL_MA5"], label="VOL_MA5")
    plt.title(f"{symbol} 成交量变化")
    plt.legend()
    plt.show()
