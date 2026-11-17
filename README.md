# 📈 Event-Driven Algorithmic Trading Engine

[![Python 3.11+](https://shields.io)](https://python.org)
[![Tests](https://shields.io)]()

A modular, institutional-grade **Event-Driven Algorithmic Trading & Backtesting Engine** built in Python, featuring strict look-ahead bias prevention, volatility-based position sizing, realistic transaction cost modeling, SQLite market data persistence, and an interactive **Streamlit** analytics dashboard.

---

## 🏗️ System Architecture

Unlike vectorized backtesters that risk data leakage, this engine processes market data sequentially through a strict chronological **Event Loop** (`MarketEvent` → `SignalEvent` → `OrderEvent` → `FillEvent`). This architecture ensures that the exact same strategy and risk logic can seamlessly transition from historical backtesting to live paper trading.

```text
┌─────────────────┐     MarketEvent     ┌──────────────────┐
│  Data Handler   │ ──────────────────► │ Strategy Engine  │
│ (SQLite/Alpaca) │                     │ (Mean Reversion) │
└─────────────────┘                     └────────┬─────────┘
         ▲                                       │ SignalEvent
         │                                       ▼
┌────────┴────────┐     FillEvent       ┌──────────────────┐
│Execution Handler│ ◄────────────────── │   Risk Manager   │
│(Sim / Paper API)│     OrderEvent      │(ATR Sizing & VaR)│
└─────────────────┘                     └──────────────────┘
```

### Key Engineering & Quantitative Features
* **Zero Look-Ahead Bias:** Signals generated on bar t are strictly executed at the opening price of bar t+1 (`next_open`), enforced via timestamp validation inside the core event dataclasses.
* **Institutional Risk Management:** Dynamic position sizing based on **Average True Range (ATR)** volatility, pre-trade exposure checks (`max_position_pct`, `max_gross_exposure`), and an automated **Drawdown Kill-Switch** that liquidates open positions if portfolio drawdown breaches the configured threshold.
* **Realistic Execution Modeling:** Incorporates unfavorable basis-point (bps) slippage and per-trade commissions to prevent over-optimistic backtest curves.
* **SQLite Persistence & Live Ready:** Automatically caches historical OHLCV bars in a local SQLite database (`data/market_data.db`) and includes modular interfaces (`typing.Protocol` & `ABC`) ready for real-time paper trading via Alpaca APIs.
* **Verified Reliability:** Covered by **64 automated unit tests** (`pytest`) validating PnL accounting, margin/cash updates, slippage asymmetry, and kill-switch triggers.

---

## 📂 Repository Structure

```text
algo-trading-engine/
├── config/
│   └── settings.yaml          # Centralized parameters (capital, costs, risk limits, strategy)
├── data/
│   └── market_data.db         # SQLite database for historical market data & logs
├── src/
│   ├── core/
│   │   ├── events.py          # Immutable dataclasses (Market, Signal, Order, Fill)
│   │   └── engine.py          # Chronological Event Loop orchestrator
│   ├── data_handler/
│   │   ├── base.py            # Abstract base data interface
│   │   ├── historical.py      # SQLite & yfinance historical bar feeder with warmup handling
│   │   └── live_alpaca.py     # Real-time paper trading data handler
│   ├── strategy/
│   │   ├── base.py            # Abstract strategy interface
│   │   └── mean_reversion.py  # Z-Score Mean Reversion with long-term trend filter
│   ├── risk/
│   │   └── risk_manager.py    # Portfolio state, ATR position sizing & drawdown kill-switch
│   ├── execution/
│   │   ├── simulated.py       # Next-open execution simulator with bps slippage & fees
│   │   └── broker_api.py      # Alpaca REST API paper execution handler
│   └── analytics/
│       └── performance.py     # Sharpe, Sortino, Calmar, Max Drawdown & Trade Log metrics
├── dashboard/
│   └── app.py                 # Interactive Streamlit & Plotly web dashboard
├── tests/
│   └── test_risk_and_pnl.py   # Pytest suite (64 unit tests)
├── main.py                    # CLI entry point for running headless backtests
├── requirements.txt           # Project dependencies
└── README.md
```

---

## 🚀 Quick Start (Local Setup)

1. **Clone the repository:**
   ```bash
   git clone https://github.com
   cd algo-trading-engine
   ```

2. **Create a virtual environment and install dependencies:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Run the headless backtest engine:**
   ```bash
   python main.py
   ```

4. **Run the automated test suite:**
   ```bash
   pytest
   ```

5. **Launch the interactive Streamlit dashboard:**
   ```bash
   python -m streamlit run dashboard/app.py
   ```

---

## 📊 Performance Analytics & Dashboard

The interactive web interface allows users to dynamically adjust initial capital, asset universe, Z-Score entry/exit thresholds, and per-trade risk limits, generating real-time visualizations for:
* **Portfolio Equity Curve** vs. Benchmark
* **Underwater Drawdown Profile**
* **Risk-Adjusted Metrics:** CAGR, Annualized Volatility, Sharpe Ratio, Sortino Ratio, Calmar Ratio, Profit Factor, and Total Execution Costs
* **Granular Trade Log** with entry/exit timestamps, slippage impact, and realized PnL

*(Tip: Add 1 or 2 screenshots of your Streamlit dashboard right here!)*

---

## ⚠️ Disclaimer
*This software is developed strictly for educational, research, and portfolio demonstration purposes. Nothing in this repository constitutes financial or investment advice.*
