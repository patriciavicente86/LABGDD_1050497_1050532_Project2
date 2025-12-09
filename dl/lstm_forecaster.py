"""
Deep Learning Demand Forecasting with LSTM (GPU accelerated)
Uses TensorFlow/Keras for time series prediction
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import yaml
import logging
from typing import Tuple, Dict
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LSTMDemandForecaster:
    """
    LSTM-based demand forecasting with GPU acceleration
    """
    
    def __init__(self, config_path: str = "env/config.yaml", use_gpu: bool = True):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Check GPU availability
        self.gpus = tf.config.list_physical_devices('GPU')
        if self.gpus and use_gpu:
            logger.info(f"GPU available: {len(self.gpus)} device(s)")
            for gpu in self.gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        else:
            logger.warning("No GPU found, using CPU")
        
        self.scaler = MinMaxScaler()
        self.model = None
    
    def load_and_prepare_data(self, data_path: str = None) -> pd.DataFrame:
        """
        Load aggregated zone-hour data from Spark output
        """
        if data_path is None:
            data_path = f"{self.config['paths']['lake']}/gold/agg_zone_hour"
        
        logger.info(f"Loading data from {data_path}")
        
        # Load parquet with pandas (smaller dataset for DL)
        df = pd.read_parquet(data_path)
        
        # Sort by time
        df = df.sort_values(['PULocationID', 'pickup_date', 'pickup_hour'])
        
        logger.info(f"Loaded {len(df)} records")
        return df
    
    def create_sequences(self, data: np.ndarray, lookback: int = 24, 
                        forecast_horizon: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM training
        lookback: number of previous timesteps to use
        forecast_horizon: number of future timesteps to predict
        """
        X, y = [], []
        
        for i in range(len(data) - lookback - forecast_horizon + 1):
            X.append(data[i:i+lookback])
            y.append(data[i+lookback:i+lookback+forecast_horizon])
        
        return np.array(X), np.array(y)
    
    def prepare_zone_data(self, df: pd.DataFrame, zone_id: int, 
                         lookback: int = 24) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare training data for a specific zone
        """
        # Filter zone data
        zone_df = df[df['PULocationID'] == zone_id].copy()
        
        # Extract features
        features = zone_df[['trip_count']].values
        
        # Normalize
        features_scaled = self.scaler.fit_transform(features)
        
        # Create sequences
        X, y = self.create_sequences(features_scaled, lookback=lookback)
        
        # Train/test split (80/20)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        logger.info(f"Zone {zone_id}: Train={len(X_train)}, Test={len(X_test)}")
        
        return X_train, X_test, y_train, y_test
    
    def build_lstm_model(self, lookback: int, n_features: int = 1) -> keras.Model:
        """
        Build LSTM model architecture
        """
        model = keras.Sequential([
            layers.LSTM(128, activation='relu', return_sequences=True, 
                       input_shape=(lookback, n_features)),
            layers.Dropout(0.2),
            layers.LSTM(64, activation='relu', return_sequences=False),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        logger.info("LSTM model built")
        logger.info(model.summary())
        
        return model
    
    def train_model(self, X_train: np.ndarray, y_train: np.ndarray,
                   X_test: np.ndarray, y_test: np.ndarray,
                   epochs: int = 50, batch_size: int = 32) -> Dict:
        """
        Train LSTM model with early stopping
        """
        logger.info("Training LSTM model...")
        
        # Build model
        self.model = self.build_lstm_model(lookback=X_train.shape[1])
        
        # Callbacks
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=10, restore_best_weights=True
        )
        
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6
        )
        
        # Train
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        return history
    
    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Evaluate model performance
        """
        logger.info("Evaluating model...")
        
        # Predictions
        y_pred = self.model.predict(X_test)
        
        # Inverse transform
        y_test_inv = self.scaler.inverse_transform(y_test.reshape(-1, 1))
        y_pred_inv = self.scaler.inverse_transform(y_pred.reshape(-1, 1))
        
        # Metrics
        rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
        mae = mean_absolute_error(y_test_inv, y_pred_inv)
        r2 = r2_score(y_test_inv, y_pred_inv)
        
        metrics = {
            "rmse": float(rmse),
            "mae": float(mae),
            "r2": float(r2),
            "predictions": y_pred_inv,
            "actual": y_test_inv
        }
        
        logger.info(f"LSTM Performance - RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.4f}")
        
        return metrics
    
    def save_model(self, output_path: str = "models/lstm_demand_forecast"):
        """
        Save trained model
        """
        os.makedirs(output_path, exist_ok=True)
        self.model.save(f"{output_path}/model.h5")
        logger.info(f"Model saved to {output_path}")
    
    def plot_predictions(self, metrics: Dict, zone_id: int, 
                        output_path: str = "reports/dl_predictions.png"):
        """
        Plot actual vs predicted values
        """
        plt.figure(figsize=(12, 6))
        
        actual = metrics["actual"].flatten()[:200]  # Plot first 200 points
        predicted = metrics["predictions"].flatten()[:200]
        
        plt.plot(actual, label='Actual', alpha=0.7)
        plt.plot(predicted, label='Predicted', alpha=0.7)
        plt.title(f'LSTM Demand Forecasting - Zone {zone_id}')
        plt.xlabel('Time')
        plt.ylabel('Trip Count')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        logger.info(f"Plot saved to {output_path}")
        plt.close()
    
    def run_pipeline(self, zone_id: int = 237, lookback: int = 24) -> Dict:
        """
        Complete DL pipeline for demand forecasting
        zone_id: specific zone to model (default 237 = Upper East Side)
        """
        logger.info(f"Running DL pipeline for zone {zone_id}")
        
        # Load data
        df = self.load_and_prepare_data()
        
        # Prepare sequences
        X_train, X_test, y_train, y_test = self.prepare_zone_data(
            df, zone_id, lookback=lookback
        )
        
        # Train model
        history = self.train_model(X_train, y_train, X_test, y_test)
        
        # Evaluate
        metrics = self.evaluate_model(X_test, y_test)
        
        # Save model
        self.save_model()
        
        # Plot
        self.plot_predictions(metrics, zone_id)
        
        logger.info("DL pipeline completed successfully")
        
        return metrics


if __name__ == "__main__":
    # Check GPU
    print("TensorFlow version:", tf.__version__)
    print("GPU available:", tf.config.list_physical_devices('GPU'))
    
    # Run pipeline
    forecaster = LSTMDemandForecaster()
    metrics = forecaster.run_pipeline(zone_id=237, lookback=24)
    
    print("\n=== LSTM Model Performance ===")
    print(f"RMSE: {metrics['rmse']:.2f}")
    print(f"MAE:  {metrics['mae']:.2f}")
    print(f"R²:   {metrics['r2']:.4f}")
