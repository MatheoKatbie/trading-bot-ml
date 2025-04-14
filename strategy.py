import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from xgboost.callback import EarlyStopping # Import EarlyStopping callback
from sklearn.model_selection import train_test_split # Or TimeSeriesSplit for better validation
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# --- Configuration --- (Can be removed as path is passed now)
# LOOKAHEAD_PERIOD = 5
# THRESHOLD_BUY = 0.01
# THRESHOLD_SELL = -0.01
# MODEL_PATH = "models" # Removed - path will be passed as argument

# --- Constants used within the module ---
LOOKAHEAD_PERIOD = 5
THRESHOLD_BUY = 0.01
THRESHOLD_SELL = -0.01

def define_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Defines the target variable (Buy/Hold/Sell) based on future returns.

    Args:
        df: DataFrame with price data (must include 'Close').

    Returns:
        DataFrame with the 'Target' column added.
    """
    df_copy = df.copy()
    df_copy['Future_Return'] = df_copy['Close'].pct_change(periods=LOOKAHEAD_PERIOD).shift(-LOOKAHEAD_PERIOD)

    conditions = [
        df_copy['Future_Return'] > THRESHOLD_BUY,
        df_copy['Future_Return'] < THRESHOLD_SELL
    ]
    choices = [2, 0]  # 2: Buy, 0: Sell
    df_copy['Target'] = np.select(conditions, choices, default=1)  # 1: Hold

    # Identify rows where Future_Return (and thus Target) is valid (not NaN)
    valid_indices = df_copy['Future_Return'].notna()
    
    # Create the final DataFrame containing only valid rows
    final_df = df_copy.loc[valid_indices].copy() # Explicit copy here

    # Now drop the intermediate column from the final DataFrame
    final_df.drop(columns=['Future_Return'], inplace=True)
    
    # Convert Target to integer type
    final_df['Target'] = final_df['Target'].astype(int)
    
    print(f"Target variable defined. Signal distribution:\n{final_df['Target'].value_counts(normalize=True)}")
    return final_df

def train_model(features: pd.DataFrame, target: pd.Series, model_params: dict = None) -> XGBClassifier | None:
    """
    Trains an XGBoost classifier.

    Args:
        features (pd.DataFrame): DataFrame containing the features (X).
        target (pd.Series): Series containing the target variable (y).
        model_params (dict, optional): Hyperparameters for XGBClassifier. Defaults to None.

    Returns:
        XGBClassifier | None: The trained XGBoost model or None if training fails.
    """
    if model_params is None:
        model_params = {
            'objective': 'multi:softmax',
            'num_class': 3, # Buy, Hold, Sell
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 3,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'eval_metric': 'mlogloss'
        }

    # Simple train/test split for demonstration.
    # !! IMPORTANT: For actual trading, use TimeSeriesSplit or Walk-Forward validation !!
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42, shuffle=False # DO NOT SHUFFLE TIME SERIES
    )

    print(f"Training data shape: {X_train.shape}, Testing data shape: {X_test.shape}")

    model = XGBClassifier(**model_params)

    # --- Diagnostic Step: Minimal fit call --- 
    # Remove eval_set, eval_metric, early_stopping_rounds to see if basic fit works
    print("[Diagnostic] Attempting minimal model.fit(X_train, y_train)...")
    try:
        # Fit with only essential arguments
        model.fit(X_train, y_train, verbose=False) 
        print("[Diagnostic] Minimal model.fit() successful.")
    except Exception as e:
        print(f"[Diagnostic] Minimal model.fit() failed: {e}")
        # If even the basic fit fails, something is fundamentally wrong
        return None # Indicate failure

    # --- Evaluate on test set AFTER fitting ---
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Training Complete (using minimal fit).")
    print(f"Test Set Accuracy: {accuracy:.4f}")
    print("Test Set Classification Report:")
    try:
        # Handle cases where not all classes (0, 1, 2) might be present in y_pred
        unique_labels = np.unique(np.concatenate((y_test, y_pred)))
        target_names_map = {0: 'Sell', 1: 'Hold', 2: 'Buy'}
        target_names_filtered = [target_names_map[label] for label in unique_labels if label in target_names_map]
        
        # Ensure labels argument matches target_names_filtered order
        labels_ordered = [label for label in unique_labels if label in target_names_map] 
        
        if len(target_names_filtered) == len(labels_ordered):
            print(classification_report(y_test, y_pred, target_names=target_names_filtered, labels=labels_ordered, zero_division=0))
        else:
             # Fallback if mapping logic fails
             print(classification_report(y_test, y_pred, zero_division=0))
    except Exception as report_error:
        print(f"[Warning] Could not generate detailed classification report: {report_error}")
        # Fallback to basic report if complex handling fails
        print(classification_report(y_test, y_pred, zero_division=0))

    return model

def save_model(model: XGBClassifier, ticker: str, model_path: str):
    """
    Saves the trained model to a file in the specified path.
    
    Args:
        model (XGBClassifier): The model to save.
        ticker (str): Ticker symbol for filename.
        model_path (str): The directory to save the model file.
    """
    os.makedirs(model_path, exist_ok=True)
    filename = os.path.join(model_path, f"{ticker}_xgb_model.joblib")
    try:
        joblib.dump(model, filename)
        print(f"Model for {ticker} saved to {filename}")
    except Exception as e:
        print(f"[Error] Failed to save model for {ticker} to {filename}: {e}")

def load_model(ticker: str, model_path: str) -> XGBClassifier | None:
    """
    Loads a trained model from a file in the specified path.
    
    Args:
        ticker (str): Ticker symbol for filename.
        model_path (str): The directory to load the model file from.

    Returns:
        XGBClassifier | None: The loaded model or None if not found/error.
    """
    filename = os.path.join(model_path, f"{ticker}_xgb_model.joblib")
    if os.path.exists(filename):
        try:
            model = joblib.load(filename)
            print(f"Model for {ticker} loaded from {filename}")
            return model
        except Exception as e:
            print(f"[Error] Failed to load model for {ticker} from {filename}: {e}")
            return None
    else:
        print(f"No pre-trained model found for {ticker} at {filename}")
        return None

def predict_signals(model: XGBClassifier, features: pd.DataFrame) -> np.ndarray:
    """
    Predicts trading signals (0: Sell, 1: Hold, 2: Buy) using the trained model.
    """
    if features.empty:
        return np.array([])
    predictions = model.predict(features)
    return predictions

# --- Example Usage --- (Requires data_loader.py and features.py)
if __name__ == '__main__':
    try:
        from data_loader import fetch_data
        from features import add_features
    except ImportError:
        print("Run this script from the project root directory or ensure data_loader.py and features.py are available.")
        raw_df = None # Cannot proceed without dependencies
    else:
        # 1. Load Data
        ticker = "AAPL"
        start = "2019-01-01" # Need enough data for features/target
        end = "2023-12-31"
        raw_df = fetch_data(ticker, start, end)

    if raw_df is not None and not raw_df.empty:
        # 2. Add Features
        df_features = add_features(raw_df.copy())

        # 3. Define Target
        df_target = define_target(df_features.copy())

        # 4. Prepare Data for Model
        # Features: Use all columns except the target and potentially OHLCV?
        # For simplicity, let's use only the calculated indicators for now.
        feature_cols = ['RSI', 'EMA_12', 'EMA_26', 'MACD', 'MACD_signal', 'MACD_diff', 'BB_high', 'BB_low', 'BB_mavg']
        X = df_target[feature_cols]
        y = df_target['Target']

        if not X.empty:
            # 5. Train Model
            print("\n--- Training Model ---")
            trained_model = train_model(X, y)

            # 6. Save Model
            print("\n--- Saving Model ---")
            save_model(trained_model, ticker, "models")

            # 7. Load Model (Demonstration)
            print("\n--- Loading Model ---")
            loaded_model = load_model(ticker, "models")

            # 8. Predict Signals (Example on last few data points)
            if loaded_model is not None:
                print("\n--- Predicting Signals (Example) ---")
                # Use the same feature set X for prediction demonstration
                # In a real scenario, you'd predict on unseen future data
                example_features = X.tail(5)
                signals = predict_signals(loaded_model, example_features)
                print(f"Features for prediction:\n{example_features}")
                print(f"Predicted signals (0=Sell, 1=Hold, 2=Buy): {signals}")
        else:
            print("Not enough data remained after feature and target calculation to train the model.")
    else:
        print("Could not load data to run the strategy example.") 