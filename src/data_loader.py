import akshare as ak
import pandas as pd
from datetime import datetime, timedelta
from src.config import START_DATE, END_DATE, INDEX_CODE

_SPOT_CACHE = {"df": None, "ts": None}
_SPOT_TTL_SECONDS = 30


def _normalize_date_str(date_str):
    """将日期格式统一为 YYYYMMDD，兼容 YYYY-MM-DD 等输入。"""
    if date_str is None:
        return None
    if isinstance(date_str, (datetime, pd.Timestamp)):
        return pd.to_datetime(date_str).strftime("%Y%m%d")
    date_str = str(date_str).strip()
    if not date_str:
        return None
    try:
        return pd.to_datetime(date_str).strftime("%Y%m%d")
    except Exception:
        return date_str.replace("-", "")


def _get_spot_df():
    now = datetime.now()
    cached_df = _SPOT_CACHE.get("df")
    cached_ts = _SPOT_CACHE.get("ts")
    if cached_df is not None and cached_ts and (now - cached_ts) < timedelta(seconds=_SPOT_TTL_SECONDS):
        return cached_df

    try:
        df = ak.stock_zh_a_spot_em()
    except Exception:
        # 兜底：如果拉取失败且缓存可用，则返回缓存，避免整条链路失败
        return cached_df if cached_df is not None else pd.DataFrame()

    _SPOT_CACHE["df"] = df
    _SPOT_CACHE["ts"] = now
    return df


def _get_realtime_quote(symbol: str):
    try:
        symbol = str(symbol).zfill(6)
        spot = _get_spot_df()
        if spot is None or spot.empty:
            return None
        row = spot.loc[spot["代码"] == symbol]
        if row.empty:
            return None
        return row.iloc[0]
    except Exception:
        return None


def get_index_constituents(index_code):
    """获取指数成分股"""
    try:
        df = ak.index_stock_cons_csindex(symbol=index_code)
    except Exception:
        return []
    if df is None or df.empty:
        return []
    code_col = "成分券代码" if "成分券代码" in df.columns else None
    if code_col is None:
        return []
    return df[code_col].astype(str).str.zfill(6).tolist()


def get_index_constituents_with_name(index_code):
    """获取指数成分股（代码 -> 名称）。"""
    try:
        df = ak.index_stock_cons_csindex(symbol=index_code)
    except Exception:
        return {}
    if df is None or df.empty:
        return {}
    code_col = "成分券代码" if "成分券代码" in df.columns else None
    name_col = "成分券名称" if "成分券名称" in df.columns else None
    if code_col is None or name_col is None:
        return {}
    codes = df[code_col].astype(str).str.zfill(6)
    names = df[name_col].astype(str)
    return dict(zip(codes, names))


def get_stock_history(symbol, use_realtime=False):
    """获取单只股票历史行情"""
    symbol = str(symbol).zfill(6)
    start = _normalize_date_str(START_DATE)
    end = _normalize_date_str(END_DATE)
    try:
        df = ak.stock_zh_a_hist(
            symbol=symbol,
            period="daily",
            start_date=start,
            end_date=end,
            adjust="qfq"
        )
    except Exception:
        return pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame()

    if "日期" in df.columns:
        df = df.rename(columns={"日期": "date"})
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
    else:
        # 如果 akshare 返回已经是索引日期，则尝试直接使用
        if df.index.name is None:
            return pd.DataFrame()
        df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)

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


def get_hs300_index(index_code: str | None = None):
    """获取指数历史行情（默认使用 INDEX_CODE）。"""
    symbol = index_code or INDEX_CODE
    start = _normalize_date_str(START_DATE)
    end = _normalize_date_str(END_DATE)
    try:
        df = ak.index_zh_a_hist(
            symbol=symbol,
            period="daily",
            start_date=start,
            end_date=end
        )
    except Exception:
        return pd.DataFrame()
    if df is None or df.empty:
        return pd.DataFrame()
    if "日期" in df.columns:
        df["日期"] = pd.to_datetime(df["日期"])
        df.set_index("日期", inplace=True)
    return df
