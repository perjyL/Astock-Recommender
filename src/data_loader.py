import akshare as ak
import pandas as pd
from datetime import datetime, timedelta
from src.config import START_DATE, END_DATE

_SPOT_CACHE = {"df": None, "ts": None}


def _get_spot_df():
    now = datetime.now()
    cached_df = _SPOT_CACHE.get("df")
    cached_ts = _SPOT_CACHE.get("ts")
    if cached_df is not None and cached_ts and (now - cached_ts) < timedelta(seconds=30):
        return cached_df

    df = ak.stock_zh_a_spot_em()
    _SPOT_CACHE["df"] = df
    _SPOT_CACHE["ts"] = now
    return df


def _get_realtime_quote(symbol: str):
    try:
        symbol = str(symbol).zfill(6)
        spot = _get_spot_df()
        row = spot.loc[spot["代码"] == symbol]
        if row.empty:
            return None
        return row.iloc[0]
    except Exception:
        return None


def get_index_constituents(index_code):
    """获取指数成分股"""
    df = ak.index_stock_cons_csindex(symbol=index_code)
    return df["成分券代码"].tolist()


def get_stock_history(symbol, use_realtime=False):
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

    if use_realtime and not df.empty:
        quote = _get_realtime_quote(symbol)
        if quote is not None:
            today = datetime.now().date()
            last_date = df.index[-1].date()
            realtime_row = {
                "开盘": quote.get("今开"),
                "收盘": quote.get("最新价"),
                "最高": quote.get("最高"),
                "最低": quote.get("最低"),
                "成交量": quote.get("成交量"),
                "成交额": quote.get("成交额"),
            }
            if last_date == today:
                for key, value in realtime_row.items():
                    if key in df.columns and pd.notna(value):
                        df.at[df.index[-1], key] = value
            elif last_date < today:
                insert_row = {col: df.iloc[-1][col] for col in df.columns}
                for key, value in realtime_row.items():
                    if key in df.columns and pd.notna(value):
                        insert_row[key] = value
                df.loc[pd.Timestamp(today)] = insert_row
                df.sort_index(inplace=True)

    return df

def get_hs300_index():
    df = ak.index_zh_a_hist(
        symbol="000300",
        period="daily",
        start_date=START_DATE,
        end_date=END_DATE
    )
    df["日期"] = pd.to_datetime(df["日期"])
    df.set_index("日期", inplace=True)
    return df
