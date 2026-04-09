---
title: Trading Environment Server
emoji: 📈
colorFrom: indigo
colorTo: green
sdk: docker
pinned: false
app_port: 8000
base_path: /
tags:
  - openenv
  - trading
  - reinforcement-learning
---

# Trading OpenEnv Environment

An RL trading environment for training LLM agents on stock trading decisions. Built on the [OpenEnv](https://github.com/meta-pytorch/OpenEnv) framework, it wraps the Alpaca MCP server with a virtual ledger for safe backtesting against historical market data.

## Features

- **Historical Backtesting** — Uses `yfinance` daily data for offline simulation across a configurable asset universe
- **Live Trading Mode** — Optional connection to the real Alpaca MCP server for paper/live trading
- **Virtual Ledger** — Tracks cash, positions, and equity in-process; validates orders against available funds
- **Shaped Reward Signal** — Equity-delta base reward with penalties for invalid trades and idle agents, plus diversification bonus
- **3 Graded Tasks** — Easy → Medium → Hard with deterministic programmatic graders (0.0–1.0)
- **Baseline Inference Script** — Uses the OpenAI API client for reproducible benchmark scores

## Quick Start

### 1. Install Dependencies

```bash
cd trading
uv sync
```

### 2. Run the Baseline Agent

```bash
export HF_TOKEN="..." 
uv run python inference.py
```

This runs all 3 tasks sequentially and prints scores.

### 3. Start the FastAPI Server

```bash
cd trading
uv run server
# Or: uv run python -m uvicorn server.app:app --reload --host 0.0.0.0 --port 8000
```

## Tasks

| # | Task | Difficulty | Steps | Objective |
|---|------|-----------|-------|-----------|
| 1 | **Capital Preservation** | Easy | 20 | Don't lose more than 5% of starting $10,000 |
| 2 | **Profitable Episode** | Medium | 50 | End the episode with positive P/L |
| 3 | **Drawdown-Controlled Alpha** | Hard | 100 | Maximize returns with max drawdown < 10% |

### Grading Criteria

Each task has a deterministic grader returning a score in [0.0, 1.0]:

- **Capital Preservation**: Linear interpolation — $9,500+ → 1.0, ≤$8,000 → 0.0
- **Profitable Episode**: 90% profit score (linear to $500) + 10% activity score (need ≥5 trades)
- **Drawdown-Controlled Alpha**: Weighted composite — 50% profit (to $1,000), 30% max drawdown control, 20% trading activity (need ≥10 trades)

## Environment Details

### Action — `TradingAction`

| Field | Type | Description |
|-------|------|-------------|
| `tool_name` | `str` | `"place_stock_order"` or `"hold"` |
| `tool_args` | `dict` | For orders: `{"symbol": "AAPL", "qty": "5", "side": "buy", "type": "market", "time_in_force": "ioc"}` |

### Observation — `TradingObservation`

| Field | Type | Description |
|-------|------|-------------|
| `result` | `list[dict]` | Tool output/content from the last action |
| `error` | `str \| None` | Error message if the last action failed |
| `market_prices` | `dict[str, float]` | Latest prices for all assets in the universe |
| `account_state` | `VirtualAccountState` | `{cash, equity, positions}` |
| `done` | `bool` | Whether the episode has terminated |
| `reward` | `float` | Shaped reward signal |

### Reward Function

The reward is calculated per step:

```
reward = (new_equity - prev_equity)                    # Base: equity delta
       - 10.0  if trade was rejected (insufficient funds/positions)
       - 5.0×n if agent holds without positions >3 consecutive times
       + 2.0   if agent holds ≥2 different assets (diversification bonus)
```

This provides signal over the full trajectory (not binary), rewards partial progress, and penalizes undesirable behavior.

### Termination Conditions

- Episode ends when `max_steps` is reached
- Episode ends early if equity drops below 80% of initial cash (margin call)

## Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `ENV_MODE` | `backtest` | `"backtest"` for historical data, `"live"` for Alpaca MCP |
| `ASSET_UNIVERSE` | `AAPL,MSFT,TSLA,NVDA,SPY` | Comma-separated list of stock symbols |
| `INITIAL_VIRTUAL_CASH` | `10000.0` | Starting cash per episode |
| `MAX_STEPS` | `100` | Maximum steps before termination |

## Project Structure

```
trading/
├── __init__.py                 # Module exports
├── README.md                   # This file
├── openenv.yaml                # OpenEnv manifest with task metadata
├── pyproject.toml              # Project metadata and dependencies
├── client.py                   # TradingEnv client (EnvClient)
├── models.py                   # Action, Observation, State Pydantic models
├── tasks/
│   ├── __init__.py             # Tasks & graders exports
│   ├── task_definitions.py     # 3 task configs (easy/medium/hard)
│   └── graders.py              # Programmatic grading functions (0.0–1.0)
└── server/
    ├── __init__.py             # Server module exports
    ├── trading_environment.py  # Core Environment implementation
    ├── simulator.py            # Historical data backtesting engine
    ├── app.py                  # FastAPI application (HTTP + WebSocket)
    └── Dockerfile              # Container image definition
```


## Development & Testing

### Direct Environment Testing

```python
from trading.server.trading_environment import TradingEnvironment
from trading.models import TradingAction

env = TradingEnvironment()
obs = env.reset(seed=42)
print(f"Cash: ${obs.account_state.cash:,.2f}")

action = TradingAction(tool_name="place_stock_order", tool_args={
    "symbol": "AAPL", "qty": "5", "side": "buy", "type": "market", "time_in_force": "ioc"
})
obs = env.step(action)
print(f"Equity: ${obs.account_state.equity:,.2f}, Reward: ${obs.reward:+,.2f}")
```
