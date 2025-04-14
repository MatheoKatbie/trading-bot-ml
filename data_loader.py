import yfinance as yf
import pandas as pd
import os

def fetch_data(ticker: str, start_date: str = None, end_date: str = None, 
                 period: str = None, interval: str = "1d",
                 data_path: str = "data") -> pd.DataFrame | None:
    """
    Fetches historical stock data (OHLCV) from yfinance or loads from cache.
    Can fetch based on start/end dates OR a period/interval.

    Args:
        ticker: Stock ticker symbol (e.g., "AAPL").
        start_date: Start date in "YYYY-MM-DD" format (used if period is None).
        end_date: End date in "YYYY-MM-DD" format (used if period is None).
        period: Period string (e.g., "1mo", "60d", "1d") - overrides start/end.
        interval: Data interval (e.g., "1m", "15m", "1d").
        data_path: Directory to store/load cached data files. 
                   If None, caching is disabled (for live fetches).

    Returns:
        pandas DataFrame with OHLCV data, or None if fetching fails.
    """
    use_cache = data_path is not None
    cache_file = None

    if use_cache:
        os.makedirs(data_path, exist_ok=True)
        # Cache filename depends on whether using dates or period
        if period is None:
             if not start_date or not end_date:
                  print("[Error] fetch_data: Must provide start_date and end_date if period is None and caching is enabled.")
                  return None
             cache_file = os.path.join(data_path, f"{ticker}_{start_date}_{end_date}_{interval}.csv")
        else:
             # Caching based on period isn't ideal as data changes, but added for completeness if needed.
             # Consider not caching period-based fetches.
             cache_file = os.path.join(data_path, f"{ticker}_period-{period}_interval-{interval}.csv")

        if cache_file and os.path.exists(cache_file):
            print(f"Loading cached data for {ticker} from {cache_file}")
            try:
                # Use header=0 (first line), index_col=0 (first column),
                # skip metadata rows 1 and 2 (0-indexed)
                data = pd.read_csv(cache_file, index_col=0, header=0, parse_dates=True, skiprows=[1, 2])
                
                # Ensure index is DatetimeIndex AFTER potentially successful parsing
                if not isinstance(data.index, pd.DatetimeIndex):
                    # This should ideally not happen now, but as a fallback:
                    print(f"[Warning] Index for {cache_file} was not parsed as DatetimeIndex. Attempting conversion.")
                    # Attempt explicit conversion, might fail if format is weird
                    data.index = pd.to_datetime(data.index, utc=True)
                
                # Ensure timezone aware for consistency (important!)
                if data.index.tz is None:
                    data.index = data.index.tz_localize('UTC')
                else:
                    data.index = data.index.tz_convert('UTC')
                # TODO: Add validation? E.g., check if expected columns exist after loading
                return data
            except Exception as e:
                print(f"Error loading cached data {cache_file}: {e}. Refetching...")

    # --- Fetch from yfinance --- 
    fetch_params = {"tickers": ticker, "interval": interval}
    if period:
        print(f"Fetching {period} of {interval} data for {ticker} (live)")
        fetch_params["period"] = period
    elif start_date and end_date:
         print(f"Fetching data for {ticker} from {start_date} to {end_date} ({interval}) interval")
         fetch_params["start"] = start_date
         fetch_params["end"] = end_date
    else:
        print("[Error] fetch_data: Must provide either start/end dates or a period.")
        return None

    try:
        data = yf.download(**fetch_params)
        if data.empty:
            print(f"No data found for {ticker} with specified parameters.")
            return None
        
        # Convert index to UTC for consistency
        if isinstance(data.index, pd.DatetimeIndex):
            if data.index.tz is None:
                data.index = data.index.tz_localize('UTC')
            else:
                data.index = data.index.tz_convert('UTC')
        
        # Save to cache only if enabled and file defined
        if use_cache and cache_file:
            data.to_csv(cache_file)
            print(f"Data for {ticker} saved to {cache_file}")
            
        return data
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return None

if __name__ == '__main__':
    # Example usage:
    ticker = "AAPL"
    start = "2020-01-01"
    end = "2023-12-31"
    df = fetch_data(ticker, start, end)

    if df is not None:
        print(f"Successfully loaded data for {ticker}:")
        print(df.head())
        print(f"Shape: {df.shape}") 