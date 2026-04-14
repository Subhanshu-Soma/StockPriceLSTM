import os
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import yfinance as yf
from dotenv import load_dotenv
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential


BASE_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def load_config() -> dict:
    load_dotenv(BASE_DIR / ".env")
    return {
        "api_key": os.getenv("YAHOO_API_KEY", "YOUR_API_KEY_HERE"),
        "symbol": os.getenv("STOCK_SYMBOL", "AAPL"),
        "start_date": os.getenv("START_DATE", "2018-01-01"),
        "end_date": os.getenv("END_DATE", "2024-01-01"),
        "lookback": int(os.getenv("LOOKBACK_WINDOW", "60")),
        "epochs": int(os.getenv("TRAIN_EPOCHS", "20")),
        "batch_size": int(os.getenv("BATCH_SIZE", "32")),
    }


def download_stock_data(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    data = yf.download(symbol, start=start_date, end=end_date, auto_adjust=True, progress=False)
    if data.empty:
        raise ValueError(
            "No stock data was returned. Check the ticker symbol, date range, and internet connection."
        )
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    required_columns = {"Close", "Open", "High", "Low", "Volume"}
    if not required_columns.issubset(set(data.columns)):
        raise ValueError(f"Downloaded data is missing required columns: {required_columns}")
    return data.reset_index()


def create_sequences(values: np.ndarray, lookback: int) -> tuple[np.ndarray, np.ndarray]:
    x_data, y_data = [], []
    for i in range(lookback, len(values)):
        x_data.append(values[i - lookback:i, 0])
        y_data.append(values[i, 0])
    x_array = np.array(x_data)
    y_array = np.array(y_data)
    x_array = x_array.reshape((x_array.shape[0], x_array.shape[1], 1))
    return x_array, y_array


def build_model(input_shape: tuple[int, int]) -> Sequential:
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


def plot_training_loss(history: tf.keras.callbacks.History, output_path: Path) -> None:
    plt.figure(figsize=(10, 6))
    plt.plot(history.history["loss"], label="Training Loss")
    if "val_loss" in history.history:
        plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("LSTM Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_predictions(dates: pd.Series, actual: np.ndarray, predicted: np.ndarray, symbol: str, output_path: Path) -> None:
    plt.figure(figsize=(12, 6))
    plt.plot(dates, actual, label="Actual Price")
    plt.plot(dates, predicted, label="Predicted Price")
    plt.title(f"{symbol} Actual vs Predicted Closing Prices")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main() -> None:
    config = load_config()
    np.random.seed(42)
    tf.random.set_seed(42)

    print(f"Loading data for {config['symbol']} from {config['start_date']} to {config['end_date']}")
    print("YAHOO_API_KEY placeholder loaded from .env for deployment compatibility.")

    stock_df = download_stock_data(
        symbol=config["symbol"],
        start_date=config["start_date"],
        end_date=config["end_date"],
    )

    close_prices = stock_df[["Close"]].copy()
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_close = scaler.fit_transform(close_prices)

    x_all, y_all = create_sequences(scaled_close, config["lookback"])
    split_index = int(len(x_all) * 0.8)

    x_train, x_test = x_all[:split_index], x_all[split_index:]
    y_train, y_test = y_all[:split_index], y_all[split_index:]

    model = build_model((x_train.shape[1], x_train.shape[2]))
    early_stopping = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

    history = model.fit(
        x_train,
        y_train,
        validation_split=0.1,
        epochs=config["epochs"],
        batch_size=config["batch_size"],
        callbacks=[early_stopping],
        verbose=1,
    )

    predictions_scaled = model.predict(x_test, verbose=0)
    predictions = scaler.inverse_transform(predictions_scaled)
    actual = scaler.inverse_transform(y_test.reshape(-1, 1))

    mae = mean_absolute_error(actual, predictions)
    rmse = np.sqrt(mean_squared_error(actual, predictions))
    mape = np.mean(np.abs((actual - predictions) / actual)) * 100

    prediction_dates = stock_df["Date"].iloc[config["lookback"] + split_index:].reset_index(drop=True)

    model.save(MODEL_DIR / "lstm_stock_model.keras")
    joblib.dump(scaler, MODEL_DIR / "lstm_stock_scaler.joblib")

    plot_training_loss(history, OUTPUT_DIR / "stock_training_loss.png")
    plot_predictions(
        prediction_dates,
        actual.flatten(),
        predictions.flatten(),
        config["symbol"],
        OUTPUT_DIR / "stock_actual_vs_predicted.png",
    )

    metrics = {
        "symbol": config["symbol"],
        "mae": float(mae),
        "rmse": float(rmse),
        "mape_percent": float(mape),
        "train_samples": int(len(x_train)),
        "test_samples": int(len(x_test)),
        "lookback_window": int(config["lookback"]),
    }
    pd.DataFrame([metrics]).to_csv(OUTPUT_DIR / "stock_metrics.csv", index=False)

    print("Training complete.")
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAPE: {mape:.2f}%")
    print(f"Model saved to: {MODEL_DIR / 'lstm_stock_model.keras'}")
    print(f"Plot saved to:  {OUTPUT_DIR / 'stock_actual_vs_predicted.png'}")


if __name__ == "__main__":
    main()
