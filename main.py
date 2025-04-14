import pandas as pd
import numpy as np
import os
import time
from datetime import datetime
from rich.console import Console
import traceback # Import traceback module

# Import project modules
from data_loader import fetch_data
from features import add_features
from strategy import define_target, train_model, load_model, save_model, predict_signals
from backtester import Backtester
from utils import plot_performance

# --- Configuration ---
CONFIG = {
    "tickers": ["AAPL", "NVDA", "INTC", "GS", "AMD", "PFE", "META", "DELL", "AMZN", "MSFT"],
    "start_date_hist": "2015-01-01", # Start date for initial historical data download (Updated)
    "train_end_date": "2023-12-31",   # End date for initial training data (Can be removed or ignored for live)
    # Live simulation settings
    "live_interval_seconds": 60 * 5, # Check every 5 minutes
    "live_hist_window": "60d",       # Fetch last 60 days for feature calculation
    "live_data_interval": "15m",     # Use 15-minute bars for live data (adjust based on availability)
    "retrain_interval_hours": 24,    # How often to retrain models (in hours)
    # Backtester settings
    "initial_capital": 10000.0,
    "transaction_cost": 0.001,
    "allocation_per_trade": 0.10,
    # Model/Feature settings
    "feature_cols": ['RSI', 'EMA_12', 'EMA_26', 'MACD', 'MACD_signal', 'MACD_diff', 'BB_high', 'BB_low', 'BB_mavg'],
    "data_path": "persistent_data", # Use persistent paths if running outside docker
    "model_path": "persistent_models"
}

console = Console()

# --- Helper function for retraining ---
def _retrain_ticker_model(ticker: str, config: dict, existing_model: object):
    """Fetches full history, recalculates features/target, retrains, and saves model."""
    console.print(f"\n[bold magenta]Attempting retraining for {ticker}...[/bold magenta]")
    try:
        # 1. Fetch full, updated daily data
        console.print(f"  Retrain Step 1: Fetching full daily history for {ticker}...")
        end_date_retrain = datetime.now().strftime('%Y-%m-%d')
        df_full_hist = fetch_data(ticker=ticker,
                                  start_date=config["start_date_hist"],
                                  end_date=end_date_retrain,
                                  interval='1d', # Use daily data for retraining history
                                  data_path=None) # Don't use cache for retrain fetch

        if df_full_hist is None or df_full_hist.empty:
            console.print(f"  [yellow]Retrain failed: Could not fetch full history for {ticker}.[/yellow]")
            return existing_model # Return the old model
        console.print(f"  Retrain Step 1: Full history fetched (Shape: {df_full_hist.shape}).")

        # 2. Add features
        console.print(f"  Retrain Step 2: Calculating features...")
        df_features = add_features(df_full_hist.copy())
        
        # !!! ADD COLUMN CLEANING LOGIC HERE (same as in live loop) !!!
        console.print(f"  Debug Initial Train: Columns BEFORE cleaning: {df_features.columns}", style="dim")
        if isinstance(df_features.columns, pd.MultiIndex):
             cleaned_columns = df_features.columns.get_level_values(0).str.strip()
        else:
             cleaned_columns = df_features.columns.str.strip()
        df_features.columns = cleaned_columns
        console.print(f"  Debug Initial Train: Columns AFTER cleaning: {df_features.columns}", style="dim")
        
        console.print(f"  Retrain Step 2: Features calculated.")

        # 3. Define target variable
        console.print(f"  Retrain Step 3: Defining target...")
        df_target = define_target(df_features.copy())
        console.print(f"  Retrain Step 3: Target defined.")

        # 4. Prepare Data for Model
        console.print(f"  Retrain Step 4: Preparing training data...")
        X_train = df_target[config["feature_cols"]]
        y_train = df_target['Target']

        if X_train.empty or y_train.empty:
            console.print(f"  [yellow]Retrain failed: Training data empty after processing for {ticker}.[/yellow]")
            return existing_model
        console.print(f"  Retrain Step 4: Training data prepared (Shape: {X_train.shape}).")

        # 5. Train Model (using minimal fit)
        console.print(f"  Retrain Step 5: Training new model...")
        new_model = train_model(X_train, y_train)

        if new_model:
            console.print(f"  [green]Retrain successful for {ticker}![/green]")
            save_model(new_model, ticker, config["model_path"])
            return new_model
        else:
            console.print(f"  [yellow]Retrain failed: Model training function returned None for {ticker}.[/yellow]")
            return existing_model

    except Exception as e:
        console.print(f"  [bold red]ERROR during retraining for {ticker}: {e}[/bold red]")
        return existing_model # Return old model on error

def run_live_simulation():
    console.print("[bold blue]=== Starting Live Trading Simulation ===[/bold blue]")

    # --- 1. Initial Historical Data Loading & Model Training ---
    console.print("\n[bold cyan]1. Loading Historical Data & Training Initial Models...[/bold cyan]")
    models = {}
    data_dict_hist = {}

    for ticker in CONFIG["tickers"]:
        console.print(f"\n--- Initial Processing: {ticker} ---")
        df_hist = fetch_data(ticker=ticker, 
                             start_date=CONFIG["start_date_hist"], 
                             end_date=CONFIG["train_end_date"], 
                             data_path=CONFIG["data_path"])
        
        if df_hist is None or df_hist.empty:
            console.print(f"[yellow]Warning: Could not load historical data for {ticker}. Skipping initial training.[/yellow]")
            continue
        data_dict_hist[ticker] = df_hist # Store for potential later use if needed

        # Add features
        try:
            df_features = add_features(df_hist.copy())
            
            # !!! ADD COLUMN CLEANING LOGIC HERE (same as in live loop) !!!
            console.print(f"  Debug Initial Train: Columns BEFORE cleaning: {df_features.columns}", style="dim")
            if isinstance(df_features.columns, pd.MultiIndex):
                 cleaned_columns = df_features.columns.get_level_values(0).str.strip()
            else:
                 cleaned_columns = df_features.columns.str.strip()
            df_features.columns = cleaned_columns
            console.print(f"  Debug Initial Train: Columns AFTER cleaning: {df_features.columns}", style="dim")
            
        except Exception as e:
             console.print(f"[yellow]Warning: Error adding features for {ticker}: {e}. Skipping training.[/yellow]")
             continue
             
        # Define target variable for training
        df_target = define_target(df_features.copy())

        # Prepare Data for Model (using all historical data up to train_end_date)
        X_train = df_target[CONFIG["feature_cols"]]
        y_train = df_target['Target']
        
        if X_train.empty or y_train.empty:
             console.print(f"[yellow]Warning: Training data is empty for {ticker} after processing. Skipping training.[/yellow]")
             continue

        # Load or Train Model
        model = load_model(ticker, CONFIG["model_path"])
        if model is None:
            console.print(f"No pre-trained model found for {ticker}. Training new model...")
            model = train_model(X_train, y_train)
            if model:
                save_model(model, ticker, CONFIG["model_path"])
            else:
                 console.print(f"[yellow]Warning: Model training failed for {ticker}. Skipping.[/yellow]")
                 continue
        
        models[ticker] = model

    if not models:
        console.print("[bold red]Error: No models available. Cannot start live simulation.[/bold red]")
        return
        
    live_tickers = list(models.keys())
    console.print(f"\nModels loaded/trained for: {live_tickers}")

    # --- 2. Initialize Backtester ---
    console.print("\n[bold cyan]2. Initializing Backtester...[/bold cyan]")
    backtester = Backtester(
        tickers=live_tickers,
        initial_capital=CONFIG["initial_capital"],
        transaction_cost=CONFIG["transaction_cost"],
        allocation_per_trade=CONFIG["allocation_per_trade"]
    )
    console.print(f"Initial Portfolio Value: ${backtester.portfolio_value:,.2f}")

    # --- 3. Live Simulation Loop ---
    console.print("\n[bold cyan]3. Starting Live Simulation Loop...[/bold cyan] (Press Ctrl+C to stop)")
    last_retrain_time = datetime.now() # Initialize last retrain time

    while True:
        try:
            current_timestamp = datetime.now()
            formatted_time = current_timestamp.strftime('%Y-%m-%d %H:%M:%S')
            console.print(f"\n[green]INFO: Starting loop iteration at {formatted_time}...[/green]")
            
            # --- Check for Periodic Retraining ---
            time_since_last_retrain = current_timestamp - last_retrain_time
            if time_since_last_retrain.total_seconds() >= CONFIG["retrain_interval_hours"] * 3600:
                console.print(f"\n[bold magenta] === Triggering Periodic Model Retraining (Interval: {CONFIG['retrain_interval_hours']}h) === [/bold magenta]")
                for ticker in live_tickers:
                    # Pass the existing model in case retraining fails
                    models[ticker] = _retrain_ticker_model(ticker, CONFIG, models.get(ticker))
                last_retrain_time = current_timestamp # Update last retrain time after attempting
                console.print(f"[bold magenta] === Periodic Retraining Cycle Complete === [/bold magenta]")
            
            console.print(f"[bold] ----- Iteration: {formatted_time} ----- [/bold]")

            latest_prices = {}
            latest_signals = {}
            fetch_errors = []

            # --- Fetch Recent Data & Predict --- 
            for ticker in live_tickers:
                try:
                    console.print(f"\n[blue]Processing ticker: {ticker}...[/blue]") # Log start
                    console.print(f"  Step 1: Fetching recent data...")
                    recent_data = fetch_data(ticker=ticker, 
                                             period=CONFIG["live_hist_window"], 
                                             interval=CONFIG["live_data_interval"], 
                                             data_path=None) 
                    console.print(f"  Step 1: Fetch successful.")

                    if recent_data is None or recent_data.empty:
                        console.print(f"  [yellow]Result: No recent data. Holding.[/yellow]")
                        latest_signals[ticker] = 1 
                        latest_prices[ticker] = np.nan
                        continue
                        
                    console.print(f"  Step 2: Getting latest price...")
                    try:
                        # Attempt to get the last scalar value using iloc[-1] then .item()
                        latest_close_price_series = recent_data['Close'].iloc[-1] # Get the last element (should be a Series if MultiIndex)
                        console.print(f"  Debug: Value from iloc[-1] for {ticker}: {latest_close_price_series} (Type: {type(latest_close_price_series)})", style="dim")
                        
                        is_valid_price = False # Reset validity flag
                        scalar_price = np.nan # Default
                        
                        try:
                             # Extract the single scalar value from the Series
                             scalar_price = latest_close_price_series.item()
                             console.print(f"  Debug: Value from .item() for {ticker}: {scalar_price} (Type: {type(scalar_price)})", style="dim")
                             
                             # Check if the scalar is NaN
                             if not pd.isna(scalar_price):
                                 try:
                                     # Ensure it's a float (might already be)
                                     latest_close_price = float(scalar_price)
                                     latest_prices[ticker] = latest_close_price
                                     console.print(f"  Step 2: Price = {latest_close_price:.2f}")
                                     is_valid_price = True # Mark as valid only if no errors
                                 except ValueError:
                                     # Log the problematic value
                                     console.print(f"  [yellow]Warning: Conversion to float failed for value '{scalar_price}' (ticker: {ticker}).[/yellow]")
                                     # is_valid_price remains False
                             else:
                                  # scalar_price was NaN
                                  console.print(f"  [yellow]Warning: Latest price value is NaN for {ticker}.[/yellow]")
                                  # is_valid_price remains False
                                  
                        except ValueError as ve_item:
                             console.print(f"  [yellow]Warning: ValueError during .item() extraction for {ticker}: {ve_item}. Series was: '{latest_close_price_series}' Holding.[/yellow]")
                             # is_valid_price remains False
                        # except AttributeError as ae_item: # Add this if .item() itself fails due to wrong type
                        #     console.print(f"  [yellow]Warning: AttributeError during .item() extraction for {ticker}: {ae_item}. Object was: '{latest_close_price_series}' (Type: {type(latest_close_price_series)}). Holding.[/yellow]")
                             # is_valid_price remains False
                             
                    except IndexError:
                        # .iloc[-1] failed, likely empty data after selection
                        console.print(f"  [yellow]Warning: Could not get last price for {ticker} (IndexError). Holding.[/yellow]")
                        is_valid_price = False
                    except Exception as price_err:
                        # Catch other potential errors during price extraction
                        console.print(f"  [yellow]Warning: Error getting price for {ticker} ({type(price_err).__name__}). Holding.[/yellow]")
                        is_valid_price = False
                        
                    if not is_valid_price:
                        console.print(f"  [yellow]Final Decision: Could not get valid latest close price for {ticker}. Holding.[/yellow]")
                        latest_prices[ticker] = np.nan
                        latest_signals[ticker] = 1
                        continue # Skip rest of processing for this ticker
                    
                    console.print(f"  Step 3: Calculating features...")
                    try:
                        features_recent = add_features(recent_data.copy())
                        
                        console.print(f"  Debug: Columns BEFORE cleaning: {features_recent.columns}", style="dim")
                        
                        # Clean column names (handle potential MultiIndex)
                        if isinstance(features_recent.columns, pd.MultiIndex):
                             # If MultiIndex, take the FIRST level (index 0) and strip whitespace
                             cleaned_columns = features_recent.columns.get_level_values(0).str.strip()
                        else:
                             # If simple Index, just strip whitespace
                             cleaned_columns = features_recent.columns.str.strip()
                        features_recent.columns = cleaned_columns # Assign cleaned columns back
                        
                        console.print(f"  Debug: Columns AFTER cleaning: {features_recent.columns}", style="dim")
                        console.print(f"  Step 3: Features calculated (columns cleaned).")
                    except ValueError as ve:
                        # Check if this is the specific ambiguity error
                        if "truth value of a Series is ambiguous" in str(ve):
                            console.print(f"  [bold red]ERROR in add_features (ValueError): {ve}[/bold red]")
                            console.print(f"  [yellow]Holding {ticker} due to feature calculation error.[/yellow]")
                            latest_signals[ticker] = 1 # Hold
                            fetch_errors.append(ticker)
                            continue # Skip to next ticker
                        else:
                            # Re-raise other ValueErrors if they are unexpected
                            console.print(f"  [bold red]Unexpected ValueError in add_features: {ve}[/bold red]")
                            raise ve # Let the outer exception handler catch it
                    except Exception as e_feat:
                         # Catch other potential errors during feature calculation
                         console.print(f"  [bold red]ERROR during add_features ({type(e_feat).__name__}): {e_feat}[/bold red]")
                         console.print(f"  [yellow]Holding {ticker} due to feature calculation error.[/yellow]")
                         latest_signals[ticker] = 1 # Hold
                         fetch_errors.append(ticker)
                         continue # Skip to next ticker
                    
                    if features_recent.empty:
                        console.print(f"  [yellow]Result: Features empty after calculation. Holding.[/yellow]")
                        latest_signals[ticker] = 1 
                        continue
                        
                    console.print(f"  Step 4: Getting latest feature row...")
                    latest_feature_row = features_recent[CONFIG["feature_cols"]].iloc[[-1]]
                    console.print(f"  Step 4: Latest feature row obtained.")

                    if latest_feature_row.isnull().values.any():
                        console.print(f"  [yellow]Result: NaN found in latest features. Holding.[/yellow]")
                        latest_signals[ticker] = 1 
                        continue
                        
                    console.print(f"  Step 5: Predicting signal...")
                    model = models[ticker]
                    signal = predict_signals(model, latest_feature_row)[0]
                    latest_signals[ticker] = signal
                    signal_text = {0:"Sell", 1:"Hold", 2:"Buy"}.get(signal, "Unknown")
                    console.print(f"  Step 5: Signal = {signal} ({signal_text})")
                    
                    # Final success message for this ticker
                    # Ensure price is valid before formatting
                    if not np.isnan(latest_prices.get(ticker, np.nan)):
                         console.print(f"  [green]Result: {ticker}: Price={latest_prices[ticker]:.2f}, Signal={signal} ({signal_text})[/green]")
                    else:
                         console.print(f"  [green]Result: {ticker}: Price=N/A, Signal={signal} ({signal_text})[/green]")

                except Exception as e:
                    # Log the error type and message more clearly
                    error_type = type(e).__name__
                    console.print(f"  [bold red]ERROR during processing for {ticker}! Type: {error_type}[/bold red]")
                    console.print(f"  [bold red]Error message: {e}[/bold red]")
                    traceback.print_exc() # Uncommented for full traceback
                    fetch_errors.append(ticker)
                    latest_signals[ticker] = 1 # Hold on error
                    latest_prices[ticker] = np.nan # Mark price unavailable

            # --- Update Backtester (Portfolio Simulation) --- 
            # Filter out tickers with unavailable prices before processing
            valid_prices = {t: p for t, p in latest_prices.items() if not np.isnan(p)}
            valid_signals = {t: s for t, s in latest_signals.items() if t in valid_prices}
            
            if not valid_prices:
                console.print("[yellow]No valid prices available for any ticker in this iteration. Skipping portfolio update.[/yellow]")
            else:
                console.print("\nUpdating simulated portfolio...")
                backtester.process_date(current_timestamp, valid_prices, valid_signals)
            
            # --- Wait for next interval ---
            console.print(f"\nSleeping for {CONFIG['live_interval_seconds']} seconds...")
            time.sleep(CONFIG['live_interval_seconds'])

        except KeyboardInterrupt:
            console.print("\n[bold yellow]KeyboardInterrupt received. Shutting down simulation...[/bold yellow]")
            break # Exit the while loop
        except Exception as loop_error:
            console.print(f"\n[bold red]Critical error in main loop: {loop_error}[/bold red]")
            console.print("Loop will pause for 60 seconds before potentially retrying...")
            time.sleep(60) # Pause before potentially catastrophic retry loop
            

    # --- 4. Final Summary (after loop ends) ---
    console.print("\n[bold cyan]4. Displaying Final Backtest Summary...[/bold cyan]")
    backtester.display_summary()

    # Optional: Generate final performance plot
    history_df = backtester.get_history()
    if not history_df.empty:
        plot_performance(history_df, filename="live_simulation_performance.png")

    console.print("[bold blue]=== Live Simulation Finished ===[/bold blue]")

if __name__ == "__main__":
    # Ensure persistent directories exist
    os.makedirs(CONFIG["data_path"], exist_ok=True)
    os.makedirs(CONFIG["model_path"], exist_ok=True)
    run_live_simulation() 