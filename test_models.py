import pandas as pd
import numpy as np
import joblib
import os
import warnings

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

try:
    from tensorflow.keras.models import load_model
    HAS_TF = True
except ImportError:
    HAS_TF = False

def main():
    print("1. Loading recent unseen data for testing...")
    train_path = r"d:\smart meter\dataset\train.csv"
    if not os.path.exists(train_path):
        print("Dataset not found.")
        return
        
    df = pd.read_csv(train_path)
    df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
    df = df.dropna(subset=['datetime'])
    df = df.sort_values('datetime')
    df = df.ffill().bfill()
    
    # Extract features identical to training
    df['hour'] = df['datetime'].dt.hour
    df['dayofweek'] = df['datetime'].dt.dayofweek
    df['month'] = df['datetime'].dt.month
    
    features = ['temperature', 'windspeed', 'hour', 'dayofweek', 'month']
    target = 'electricity_consumption'
    
    # Take the last 15 rows as test samples
    X_test_sample = df[features].iloc[-15:]
    y_test_true = df[target].iloc[-15:]
    
    print("\n2. Loading trained models...")
    scaler_path = r'd:\smart meter\scaler_X.pkl'
    if not os.path.exists(scaler_path):
        print(f"Scaler not found at {scaler_path}. Please run train_models.py first.")
        return
        
    scaler_X = joblib.load(scaler_path)
    X_test_scaled = scaler_X.transform(X_test_sample)
    
    model_names = ["Linear Regression", "Random Forest", "SVR", "XGBoost"]
    predictions = {}
    
    for name in model_names:
        model_path = f'd:\\smart meter\\{name.replace(" ", "_")}_model.pkl'
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            preds = model.predict(X_test_scaled)
            predictions[name] = preds
        else:
            print(f"Model {name} not found. Skipping.")
            
    if HAS_TF:
        lstm_path = r'd:\smart meter\LSTM_model.h5'
        if os.path.exists(lstm_path):
            lstm_model = load_model(lstm_path, compile=False)
            X_test_lstm = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))
            lstm_preds = lstm_model.predict(X_test_lstm, verbose=0).flatten()
            predictions['LSTM'] = lstm_preds
        else:
            print("LSTM model not found. Skipping.")
            
    print("\n3. Testing predictions on a 15-hour sequence:")
    print("-" * 110)
    
    result_df = pd.DataFrame({
        'Date/Time': df['datetime'].iloc[-15:].dt.strftime('%Y-%m-%d %H:%M').values,
        'Actual': y_test_true.values
    })
    
    for name, preds in predictions.items():
        result_df[f'{name}'] = preds
        
    print(result_df.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print("-" * 110)

if __name__ == "__main__":
    main()
