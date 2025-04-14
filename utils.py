import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_performance(history_df: pd.DataFrame, filename: str = "performance_plot.png"):
    """
    Plots the portfolio value over time from the backtest history.
    Saves the plot to a file.

    Args:
        history_df (pd.DataFrame): DataFrame returned by backtester.get_history(). 
                                   Must have a 'Portfolio Value' column and a DatetimeIndex.
        filename (str): The name of the file to save the plot to.
    """
    if history_df is None or history_df.empty:
        print("[Warning] History DataFrame is empty. Cannot generate performance plot.")
        return

    if 'Portfolio Value' not in history_df.columns:
        print("[Warning] 'Portfolio Value' column not found in history. Cannot plot performance.")
        return
        
    if not isinstance(history_df.index, pd.DatetimeIndex):
         print("[Warning] History DataFrame index is not a DatetimeIndex. Cannot plot performance.")
         # Attempt to convert if possible, otherwise return
         try:
             history_df.index = pd.to_datetime(history_df.index)
         except Exception:
             return

    plt.style.use('seaborn-v0_8-darkgrid') # Use a nice style
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(history_df.index, history_df['Portfolio Value'], label='Portfolio Value', color='blue')
    
    ax.set_title('Portfolio Performance Over Time', fontsize=16)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Portfolio Value ($)', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True)
    
    # Format y-axis to show currency
    try:
        from matplotlib.ticker import FuncFormatter
        formatter = FuncFormatter(lambda y, _: f'${y:,.0f}')
        ax.yaxis.set_major_formatter(formatter)
    except ImportError:
        print("[Info] Could not import FuncFormatter for currency formatting.")

    plt.xticks(rotation=45)
    plt.tight_layout() # Adjust layout to prevent labels overlapping

    try:
        plt.savefig(filename)
        print(f"Performance plot saved to {filename}")
    except Exception as e:
        print(f"[Error] Failed to save performance plot: {e}")
    
    # Optionally display the plot if running in an interactive environment
    # plt.show()

# Example usage (if you want to test this file directly)
if __name__ == '__main__':
    # Create dummy data similar to backtester output
    dates = pd.date_range(start='2023-01-01', periods=100, freq='B')
    values = 10000 * (1 + (0.001 * pd.Series(range(100)) + 0.005 * pd.Series(pd.np.random.randn(100)).cumsum()))
    dummy_history = pd.DataFrame({'Portfolio Value': values}, index=dates)
    
    print("Generating plot with dummy data...")
    plot_performance(dummy_history, "dummy_performance_plot.png")
    print("Check for dummy_performance_plot.png in the current directory.") 