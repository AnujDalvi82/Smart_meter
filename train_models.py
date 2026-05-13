import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import os
import warnings
import joblib

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("XGBoost not found. Please install via 'pip install xgboost'.")

try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense
    from tensorflow.keras.callbacks import EarlyStopping
    HAS_TF = True
except ImportError:
    HAS_TF = False
    print("TensorFlow not found. Please install via 'pip install tensorflow'.")

def calculate_accuracy(y_true, y_pred):
    """Calculate a pseudo-accuracy for regression based on MAPE."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # Avoid division by zero
    mask = y_true != 0
    if np.sum(mask) == 0:
        return 0.0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    accuracy = max(0, 100 - mape)
    return accuracy

def main():
    train_path = r"d:\smart meter\dataset\train.csv"
    gui_path = r"d:\smart meter\dataset\electricity.gui"
    
    if not os.path.exists(train_path) or not os.path.exists(gui_path):
        print("Error: Datasets not found.")
        return

    print("1. Loading and Preprocessing Data...")
    # Load first dataset
    df_train = pd.read_csv(train_path)
    
    # Load second dataset (electricity.gui is a CSV despite its extension)
    df_gui = pd.read_csv(gui_path)
    
    # Preprocess electricity.gui
    # Replace '?' with NaN
    df_gui = df_gui.replace('?', np.nan)
    
    # Rename columns to match train.csv
    df_gui = df_gui.rename(columns={
        'DateTime': 'datetime',
        'ORKTemperature': 'temperature',
        'ORKWindspeed': 'windspeed',
        'SystemLoadEP2': 'electricity_consumption'
    })
    
    # Select common columns
    common_cols = ['datetime', 'temperature', 'windspeed', 'electricity_consumption']
    
    df1 = df_train[common_cols].copy()
    df2 = df_gui[common_cols].copy()
    
    # Ensure numeric types
    for col in ['temperature', 'windspeed', 'electricity_consumption']:
        df1[col] = pd.to_numeric(df1[col], errors='coerce')
        df2[col] = pd.to_numeric(df2[col], errors='coerce')
        
    # Merge both datasets
    print("  -> Merging datasets...")
    df = pd.concat([df1, df2], ignore_index=True)
    
    # Convert datetime and sort
    df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
    df = df.dropna(subset=['datetime'])
    df = df.sort_values('datetime')
    
    # Handle missing values
    df = df.ffill().bfill()
    
    # Extract temporal features
    df['hour'] = df['datetime'].dt.hour
    df['dayofweek'] = df['datetime'].dt.dayofweek
    df['month'] = df['datetime'].dt.month
    
    features = ['temperature', 'windspeed', 'hour', 'dayofweek', 'month']
    target = 'electricity_consumption'
    
    X = df[features]
    y = df[target]
    
    # Time series Train-test split (80% train, 20% test)
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # Scale features
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    
    # Save the scaler
    joblib.dump(scaler_X, r'd:\smart meter\scaler_X.pkl')
    
    # Traditional Models Setup
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1),
        "SVR": SVR(kernel='rbf', C=10.0, gamma='scale', max_iter=500) # Limited iter for speed
    }
    if HAS_XGB:
        models["XGBoost"] = xgb.XGBRegressor(n_estimators=50, random_state=42)
    
    results = {}
    predictions = {}
    
    print("\n2. Training Traditional Models...")
    for name, model in models.items():
        print(f"  -> Training {name}...")
        model.fit(X_train_scaled, y_train)
        
        # Save model
        joblib.dump(model, f'd:\\smart meter\\{name.replace(" ", "_")}_model.pkl')
        
        preds = model.predict(X_test_scaled)
        predictions[name] = preds
        
        acc = calculate_accuracy(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        
        results[name] = {'Accuracy': acc, 'RMSE': rmse, 'MAE': mae, 'R2': r2}
        
    # LSTM Prep and Training
    if HAS_TF:
        print("\n3. Training LSTM Model...")
        X_train_lstm = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
        X_test_lstm = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))
        
        lstm_model = Sequential([
            LSTM(64, activation='relu', input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])),
            Dense(32, activation='relu'),
            Dense(1)
        ])
        lstm_model.compile(optimizer='adam', loss='mse')
        
        early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        
        print("  -> Fitting LSTM...")
        lstm_model.fit(X_train_lstm, y_train, epochs=10, batch_size=64, 
                       validation_data=(X_test_lstm, y_test), 
                       callbacks=[early_stop], verbose=0)
        
        # Save LSTM model
        lstm_model.save(r'd:\smart meter\LSTM_model.h5')
        
        lstm_preds = lstm_model.predict(X_test_lstm, verbose=0).flatten()
        predictions['LSTM'] = lstm_preds
        
        lstm_acc = calculate_accuracy(y_test, lstm_preds)
        lstm_rmse = np.sqrt(mean_squared_error(y_test, lstm_preds))
        lstm_mae = mean_absolute_error(y_test, lstm_preds)
        lstm_r2 = r2_score(y_test, lstm_preds)
        
        results['LSTM'] = {'Accuracy': lstm_acc, 'RMSE': lstm_rmse, 'MAE': lstm_mae, 'R2': lstm_r2}

    print("\n--- Model Comparison Performance ---")
    print(f"{'Model':<20} | {'Accuracy (%)':<12} | {'RMSE':<10} | {'MAE':<10} | {'R2 Score':<10}")
    print("-" * 72)
    for name, metrics in results.items():
        print(f"{name:<20} | {metrics['Accuracy']:<12.2f} | {metrics['RMSE']:<10.2f} | {metrics['MAE']:<10.2f} | {metrics['R2']:<10.2f}")
        
    print("\n4. Plotting Actual vs Predicted...")
    plt.figure(figsize=(15, 6))
    
    subset_len = min(500, len(y_test)) # Plot subset for clarity
    plt.plot(y_test.values[:subset_len], label='Actual', color='black', linewidth=2.5)
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    for i, (name, preds) in enumerate(predictions.items()):
        plt.plot(preds[:subset_len], label=f'{name} Preds', linestyle='--', alpha=0.8, color=colors[i % len(colors)])
        
    plt.title('Energy Consumption: Actual vs Predicted')
    plt.xlabel('Time (Test Set Subset)')
    plt.ylabel('Electricity Consumption')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plot_path = r"d:\smart meter\model_comparison_plot.png"
    plt.savefig(plot_path)
    print(f"\nPlot successfully saved to: {plot_path}")

if __name__ == "__main__":
    main()
