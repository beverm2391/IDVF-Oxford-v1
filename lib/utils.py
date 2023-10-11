import pandas_market_calendars as mcal
import pandas as pd
from typing import List, Tuple
import os

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

def get_dfs(n=93):
    """Returns a list of n dfs from the oxford-93-5yrs-minute dataset."""

    DATA_DIR = "/Users/beneverman/Documents/Coding/bp-quant/shared_data/POLYGON/oxford-93-5yrs-minute"
    def _paths():
        paths = [p for p in os.listdir(DATA_DIR) if p.endswith(".csv")]
        assert len(paths) == 93, "Expected 93 paths, got {}".format(len(paths))
        return paths
    
    dfs = [pd.read_csv(os.path.join(DATA_DIR, p)) for p in _paths()[:n]]
    return dfs