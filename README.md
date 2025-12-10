# HMM-regime-DQN-trader-Peru

Agente de trading algorítmico con HMM para detectar regímenes de mercado y Deep Q-Network (DQN) para decidir Buy/Sell/Hold usando múltiples fuentes (APIs + Excel). Incluye backtesting y métricas Sharpe/Drawdown.

## 📋 Overview

This project implements a Reinforcement Learning (RL) trading agent that combines:
- **Hidden Markov Models (HMM)** for market regime detection (bull, bear, sideways)
- **Deep Q-Network (DQN)** for learning optimal trading strategies
- **Multiple data sources**: Yahoo Finance (yfinance), FRED API, Excel files, and sentiment analysis
- **Comprehensive backtesting** with performance metrics (Sharpe ratio, cumulative returns, maximum drawdown)

## 🏗️ Project Structure

```
HMM-regime-DQN-trader-Peru/
├── src/
│   ├── __init__.py
│   ├── ingest.py          # Data ingestion (yfinance, FRED, Excel, sentiment)
│   ├── preprocess.py      # Data preprocessing and feature engineering
│   ├── hmm_regimes.py     # HMM regime detection (3 states: bull, bear, sideways)
│   ├── trading_env.py     # Gymnasium trading environment (Buy/Sell/Hold + fees)
│   └── dqn.py            # DQN agent (replay buffer, epsilon-greedy, target network)
├── tests/
│   ├── test_ingest.py     # Tests for data ingestion
│   ├── test_preprocess.py # Tests for preprocessing
│   ├── test_hmm_regimes.py # Tests for HMM
│   ├── test_trading_env.py # Tests for trading environment
│   └── test_dqn.py        # Tests for DQN agent
├── train.py               # Training script
├── backtest.py            # Backtesting and comparison script
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## 🚀 Setup

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/Salvador0302/HMM-regime-DQN-trader-Peru.git
cd HMM-regime-DQN-trader-Peru
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. (Optional) Set up FRED API key for economic indicators:
```bash
export FRED_API_KEY="your_fred_api_key_here"
```

You can get a free FRED API key from: https://fred.stlouisfed.org/docs/api/api_key.html

## 📖 Usage

### Training the Agent

#### Train with HMM regime detection (recommended):
```bash
python train.py --ticker AAPL --start-date 2020-01-01 --end-date 2023-12-31 --episodes 100 --use-hmm
```

#### Train without HMM regime detection:
```bash
python train.py --ticker AAPL --start-date 2020-01-01 --end-date 2023-12-31 --episodes 100 --no-hmm
```

#### Training Parameters:
- `--ticker`: Stock ticker symbol (default: AAPL)
- `--start-date`: Start date in YYYY-MM-DD format (default: 2020-01-01)
- `--end-date`: End date in YYYY-MM-DD format (default: 2023-12-31)
- `--episodes`: Number of training episodes (default: 100)
- `--use-hmm`: Enable HMM regime detection (default: True)
- `--no-hmm`: Disable HMM regime detection
- `--save-models`: Save trained models (default: True)

The training script will:
1. Fetch historical stock data from Yahoo Finance
2. Preprocess data and add technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands, etc.)
3. Detect market regimes using HMM (if enabled)
4. Train the DQN agent using the trading environment
5. Save trained models to `models/` directory
6. Generate training plots in `results/` directory

### Backtesting and Comparison

After training both models (with and without HMM), run backtesting:

```bash
python backtest.py --ticker AAPL --start-date 2020-01-01 --end-date 2023-12-31
```

The backtesting script will:
1. Load both trained models
2. Run backtest on the test dataset
3. Calculate performance metrics:
   - **Sharpe Ratio**: Risk-adjusted returns
   - **Cumulative Returns**: Total portfolio return
   - **Maximum Drawdown**: Worst peak-to-trough decline
   - **Win Rate**: Percentage of profitable trades
4. Compare with buy-and-hold baseline
5. Generate comparison visualizations in `results/` directory

### Running Tests

Run all unit tests:
```bash
python -m pytest tests/
```

Run specific test module:
```bash
python -m pytest tests/test_dqn.py -v
```

Or using unittest:
```bash
python -m unittest discover tests/
```

## 🔧 Components

### 1. Data Ingestion (`src/ingest.py`)
- **Yahoo Finance**: Fetches OHLCV data for stocks
- **FRED API**: Retrieves economic indicators (GDP, unemployment, interest rates)
- **Excel Import**: Loads data from Excel files
- **Sentiment Placeholder**: Framework for sentiment analysis integration

### 2. Data Preprocessing (`src/preprocess.py`)
- Technical indicators: SMA, EMA, RSI, MACD, Bollinger Bands
- Feature engineering: Returns, volatility, lag features
- Data normalization using StandardScaler
- Train/test splitting

### 3. HMM Regime Detection (`src/hmm_regimes.py`)
- Gaussian Hidden Markov Model with 3 states
- Automatically characterizes regimes based on returns and volatility
- Labels states as: bull, bear, sideways
- Provides state transition probabilities

### 4. Trading Environment (`src/trading_env.py`)
- Gymnasium-compatible environment
- **Actions**: 0=Hold, 1=Buy, 2=Sell
- **Observations**: Market features + position information
- **Rewards**: Based on portfolio value changes
- Transaction fees simulation
- Position tracking and portfolio management

### 5. DQN Agent (`src/dqn.py`)
- **Neural Network**: Deep Q-Network with 3 hidden layers (128, 64, 32 neurons)
- **Experience Replay**: Buffer with 10,000 capacity
- **Epsilon-Greedy Exploration**: Decaying from 1.0 to 0.01
- **Target Network**: Updated every 10 training steps
- **Optimizer**: Adam with learning rate 0.001

## 📊 Performance Metrics

The system evaluates trading strategies using:

1. **Sharpe Ratio**: Measures risk-adjusted returns
   - Formula: `(Mean Return - Risk-Free Rate) / Standard Deviation of Returns * √252`
   - Higher is better (>1 is good, >2 is excellent)

2. **Cumulative Returns**: Total percentage return over the period
   - Formula: `(Final Value - Initial Value) / Initial Value * 100`

3. **Maximum Drawdown**: Largest peak-to-trough decline
   - Formula: `Min((Portfolio Value - Cumulative Max) / Cumulative Max)`
   - Lower absolute value is better

4. **Win Rate**: Percentage of profitable trades
   - Formula: `Winning Trades / Total Trades * 100`

## 🎯 Example Workflow

Complete workflow for training and evaluation:

```bash
# 1. Train agent WITH HMM
python train.py --ticker AAPL --episodes 100 --use-hmm

# 2. Train agent WITHOUT HMM
python train.py --ticker AAPL --episodes 100 --no-hmm

# 3. Compare performance
python backtest.py --ticker AAPL

# 4. Run tests
python -m pytest tests/ -v
```

## 📈 Expected Output

### Training Output:
```
======================================================================
Training DQN Agent WITH HMM Regime Detection
Ticker: AAPL | Period: 2020-01-01 to 2023-12-31
======================================================================

Step 1: Ingesting data...
  ✓ Fetched 1008 rows of data

Step 2: Preprocessing data...
  ✓ Created 45 features
  ✓ 958 rows after cleaning

Step 3: Detecting HMM regimes...
============================================================
REGIME SUMMARY
============================================================

BULL (State 0):
  Mean Return: 0.0025
  Volatility:  0.0150
  Occurrences: 320 (33.40%)
...
```

### Backtest Output:
```
======================================================================
BACKTESTING COMPARISON: WITH vs WITHOUT HMM
Ticker: AAPL | Period: 2020-01-01 to 2023-12-31
======================================================================

Running backtest: DQN WITH HMM
------------------------------------------------------------
  Final Portfolio Value: $12,450.00
  Cumulative Return: 24.50%
  Sharpe Ratio: 1.2500
  Max Drawdown: -8.50%
  Total Trades: 45
  Winning Trades: 28 (62.22%)
...
```

## 🔬 Advanced Usage

### Custom Feature Engineering

Modify `src/preprocess.py` to add custom technical indicators:

```python
def add_custom_indicator(self, df):
    # Add your custom indicator
    df['custom_indicator'] = ...
    return df
```

### Custom Regime Detection

Adjust HMM parameters in `src/hmm_regimes.py`:

```python
detector = HMMRegimeDetector(
    n_states=4,  # Change number of states
    random_state=42
)
```

### Environment Customization

Modify trading rules in `src/trading_env.py`:

```python
env = TradingEnv(
    initial_balance=50000,  # Change initial capital
    transaction_fee=0.002,   # Adjust fees
    max_position=10          # Allow multiple positions
)
```

## 🐛 Troubleshooting

### Issue: TensorFlow GPU not found
```bash
# Install TensorFlow with GPU support
pip install tensorflow[and-cuda]
```

### Issue: FRED API key error
```bash
# Set environment variable
export FRED_API_KEY="your_key_here"
```

### Issue: Yahoo Finance download fails
```bash
# Update yfinance
pip install --upgrade yfinance
```

## 📝 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or suggestions, please open an issue on GitHub.

## 🙏 Acknowledgments

- **hmmlearn**: Hidden Markov Models implementation
- **Gymnasium**: RL environment framework
- **TensorFlow/Keras**: Deep learning framework
- **yfinance**: Yahoo Finance data API
- **FRED**: Federal Reserve Economic Data API
