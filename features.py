import pandas as pd
import ta

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds technical indicators and ML features to the DataFrame.

    Args:
        df: DataFrame with OHLCV data.

    Returns:
        DataFrame with added features.
    """
    # Ensure required columns exist
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Input DataFrame must contain columns: {required_cols}")

    # Calculate Technical Indicators using 'ta' library
    # RSI
    df['RSI'] = ta.momentum.rsi(df['Close'].squeeze())

    # EMA
    df['EMA_12'] = ta.trend.ema_indicator(df['Close'].squeeze(), window=12)
    df['EMA_26'] = ta.trend.ema_indicator(df['Close'].squeeze(), window=26)

    # MACD
    macd = ta.trend.MACD(df['Close'].squeeze())
    df['MACD'] = macd.macd()
    df['MACD_signal'] = macd.macd_signal()
    df['MACD_diff'] = macd.macd_diff()

    # Bollinger Bands
    bollinger = ta.volatility.BollingerBands(df['Close'].squeeze())
    df['BB_high'] = bollinger.bollinger_hband()
    df['BB_low'] = bollinger.bollinger_lband()
    df['BB_mavg'] = bollinger.bollinger_mavg() # Middle Band

    # Add more features as needed (e.g., lagged returns, volatility)

    # Drop rows with NaN values created by indicators with windows
    # Ensure enough historical data is available before dropping NaNs
    initial_len = len(df)
    df.dropna(inplace=True)
    print(f"Dropped {initial_len - len(df)} rows with NaN values after feature calculation.")

    return df

if __name__ == '__main__':
    # Example Usage (requires data_loader.py)
    try:
        from data_loader import fetch_data
    except ImportError:
        print("Run this script from the project root directory or ensure data_loader.py is in the PYTHONPATH.")
        # Create dummy data for demonstration if data_loader is not found
        dates = pd.date_range(start='2023-01-01', periods=100, freq='B')
        data = {
            'Open': [100 + i + 10 * (0.5 - pd.np.random.rand()) for i in range(100)],
            'High': [100 + i + 15 * pd.np.random.rand() for i in range(100)],
            'Low': [100 + i - 15 * pd.np.random.rand() for i in range(100)],
            'Close': [100 + i + 10 * (0.5 - pd.np.random.rand()) for i in range(100)],
            'Volume': [1000000 + i * 1000 for i in range(100)]
        }
        raw_df = pd.DataFrame(data, index=dates)
        raw_df['Adj Close'] = raw_df['Close'] # Add Adj Close if needed
    else:
        # Fetch real data if data_loader is available
        ticker = "AAPL"
        start = "2020-01-01"
        end = "2023-12-31"
        raw_df = fetch_data(ticker, start, end)

    if raw_df is not None and not raw_df.empty:
        df_with_features = add_features(raw_df.copy()) # Use copy to avoid modifying original df
        print("\nDataFrame with features:")
        print(df_with_features.head())
        print(f"\nShape after adding features: {df_with_features.shape}")
        print("\nColumns:", df_with_features.columns.tolist())
    else:
        print("Could not load or generate data for feature calculation example.") 