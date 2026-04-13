# StockPriceLSTM
A project that determines stock prices using LSTM
## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows
pip install -r requirements.txt
cp .env.example .env
```
Run:

```bash
python stock_price_prediction_lstm/src/train_lstm.py
```

What it does:
- Downloads historical stock data using Yahoo Finance
- Scales close prices with MinMaxScaler
- Builds time sequences for LSTM training
- Trains an LSTM model in TensorFlow/Keras
- Evaluates on a holdout set
- Saves plots and the trained model

Outputs:
- `outputs/stock_actual_vs_predicted.png`
- `outputs/stock_training_loss.png`
- `models/lstm_stock_model.keras`
