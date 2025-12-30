import akshare as ak
import pandas as pd
from config import START_DATE, END_DATE


def get_index_constituents(index_code):
    """获取指数成分股"""
    df = ak.index_stock_cons_csindex(symbol=index_code)
    return df["成分券代码"].tolist()


def get_stock_history(symbol):
    """获取单只股票历史行情"""
    df = ak.stock_zh_a_hist(
        symbol=symbol,
        period="daily",
        start_date=START_DATE,
        end_date=END_DATE,
        adjust="qfq"
    )
    df = df.rename(columns={"日期": "date"})
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    return df
