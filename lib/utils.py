import pandas_market_calendars as mcal
import pandas as pd
from typing import List, Tuple
import os
import numpy as np

# ! Market Utils ===============================================================

def get_nyse_calendar(start: str, end: str) -> pd.DataFrame:
    """Returns a dataframe of the NYSE calendar."""
    nyse = mcal.get_calendar('NYSE')
    return nyse.schedule(start_date=start, end_date=end)

def get_nyse_date_tups(start: str, end: str = 'today', unix=False) -> List[Tuple[str, str]]:
    """
    Get a list of tuples of (open, close) datetimes for NYSE trading days between start and end dates. Updated to optionally ouput unix timestamps.
    """
    if end == 'today': end = pd.Timestamp.now().strftime('%Y-%m-%d') # get today! 
    assert pd.Timestamp(start) < pd.Timestamp(end), "start date must be before end date"

    nyse = get_nyse_calendar(start, end) # get nyse calendar

    decode_str = "%Y-%m-%d"
    to_str = lambda x: pd.to_datetime(x, utc=True).tz_convert('America/New_York').strftime(decode_str) # convert to nyse tz, get string
    to_unix = lambda x: int(pd.to_datetime(x, utc=True).tz_convert('America/New_York').timestamp() * 1000) # convert to nyse tz, get unix timestamp

    if unix:
        tups = [(to_unix(a), to_unix(b)) for a, b in zip(nyse['market_open'], nyse['market_close'])] # make unix tups from open/close
    else:
        tups = [(to_str(a), to_str(b)) for a, b in zip(nyse['market_open'], nyse['market_close'])] # make string tups from open/close

    assert tups is not None and len(tups) > 0, "tups must be non-empty. you probably provided dates that are not NYSE trading days."
    return tups

# ! Data Utils ==================================================================

def get_oxford_dfs(n=93):
    """Returns a list of n dfs from the oxford-93-5yrs-minute dataset."""

    assert n > 0 and n <= 93, "n must be between 1 and 93"

    DATA_DIR = "/Users/beneverman/Documents/Coding/bp-quant/shared_data/POLYGON/oxford-93-5yrs-minute"
    def _paths():
        paths = [p for p in os.listdir(DATA_DIR) if p.endswith(".csv")]
        assert len(paths) == 93, "Expected 93 paths, got {}".format(len(paths))
        return paths
    
    dfs = [pd.read_csv(os.path.join(DATA_DIR, p), index_col=0) for p in _paths()[:n]]
    if len(dfs) == 1: return dfs[0] # if only one df, return it directly (not in a list)
    return dfs

def get_65min_aggs():
    DATA_DIR = "/Users/beneverman/Documents/Coding/QuantHive/IDVF-Oxford-v1/data/processed-5yr-93-minute/65min_aggs.csv"
    return pd.read_csv(DATA_DIR, index_col=0)

def get_65min_rv():
    DATA_DIR = "/Users/beneverman/Documents/Coding/QuantHive/IDVF-Oxford-v1/data/processed-5yr-93-minute/65min_rv.csv"
    return pd.read_csv(DATA_DIR, index_col=0)

# ! Stats Utils =================================================================

def rv(series: pd.Series, window: int) -> pd.Series:
    """
    Realized volatility is defined in [Volatility Forecasting with Machine Learning
    and Intraday Commonality](https://arxiv.org/pdf/2202.08962.pdf) as:

    $$RV_{i,t}(h)=\log(\sum_{s=t-h+1}^{t}r^2_{i,s})$$
    """
    assert window > 0, "Window must be greater than 0"
    fuzz = 1e-16
    # returns = series.pct_change() # returns
    log_returns = np.log(series / series.shift(1)) # log returns
    squared_returns = log_returns**2 # squared returns
    sum_of_squares = squared_returns.rolling(window=window).sum() # sum of squared returns
    rv = np.log(sum_of_squares + fuzz) # log of sum of squared returns
    assert rv.isna().sum() == window, "RV should have NaNs at the beginning" # ? should have one nan from logret and window - 1 from rolling = window
    return rv

def rv_single_window(x: pd.Series):
    """RV over a single window/bin of time"""
    log_returns = np.log(x / x.shift(1))
    squared_log_returns = log_returns ** 2
    rv = np.log(squared_log_returns.sum() + 1e-16)
    return rv