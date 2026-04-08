# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Trading Environment Implementation.

An OpenEnv environment that wraps the Alpaca MCP server.
"""

import asyncio
import json
import os
import queue
import sys
import threading
from typing import Any, Dict, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

from trading.models import TradingAction, TradingObservation, VirtualAccountState
from trading.server.simulator import SimulatedBroker
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class MCPClientManager:
    """Manages the lifecycle of the async MCP client within a synchronous Environment."""
    
    def __init__(self):
        self.is_ready = threading.Event()
        self.loop: asyncio.AbstractEventLoop | None = None
        self.session = None
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()
        
        # Wait for initialization to complete
        self.is_ready.wait(timeout=30)
        if not self.is_ready.is_set():
            raise RuntimeError("MCP client failed to initialize within 30 seconds.")

    def _run_loop(self):
        if sys.platform.startswith("win"):
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
            
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self._async_main())

    async def _async_main(self):
        env = os.environ.copy()
        try:
            from dotenv import load_dotenv
            load_dotenv()
            # Reload env after dotenv
            for k, v in os.environ.items():
                env[k] = v
        except ImportError:
            pass
            
        # Fallback to prevent immediate crash if keys missing but still required by MCP
        if "ALPACA_API_KEY" not in env:
            env["ALPACA_API_KEY"] = "dummy_api_key"
        if "ALPACA_SECRET_KEY" not in env:
            env["ALPACA_SECRET_KEY"] = "dummy_secret_key"
            
        # Default to paper trade if not explicitly false to ensure safety
        if env.get("ALPACA_PAPER_TRADE", "").lower() != "false":
            env["ALPACA_PAPER_TRADE"] = "true"
            
        # Enforce toolset limits to prevent hallucinations
        if "ALPACA_TOOLSETS" not in env:
            # env["ALPACA_TOOLSETS"] = "account,trading,assets,stock-data"
            env["ALPACA_TOOLSETS"] = "account,trading,stock-data"
            
        server_params = StdioServerParameters(
            command="uvx",
            args=["alpaca-mcp-server"],
            env=env
        )
        
        try:
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    self.session = session
                    self.is_ready.set()
                    
                    # Keep the connection open
                    while True:
                        await asyncio.sleep(1)
                                
        except Exception as e:
            print(f"MCP Background Thread Error: {e}")
            self.session = None
            self.is_ready.set()  # Prevent blocking forever if failed

    def call_tool(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Synchronously request a tool call and wait for the result."""
        if not self.session or not self.loop:
            return {"success": False, "error": "MCP Session not initialized"}
            
        async def do_call():
            try:
                result = await self.session.call_tool(name, arguments=args)
                content_list = []
                for content in result.content:
                    if content.type == "text":
                        content_list.append({"text": content.text})
                    else:
                        content_list.append({"content": str(content)})
                return {"success": True, "result": content_list}
            except Exception as e:
                return {"success": False, "error": str(e)}

        try:
            future = asyncio.run_coroutine_threadsafe(do_call(), self.loop)
            return future.result(timeout=15)
        except Exception as e:
            return {"success": False, "error": f"Timeout or execution error: {e}"}


class TradingEnvironment(Environment):
    """
    An environment for training RL trading agents using the Alpaca MCP server.
    """

    # Set to False because MCP client spins up a heavy process natively
    SUPPORTS_CONCURRENT_SESSIONS: bool = False

    def __init__(self):
        """Initialize the trading environment."""
        super().__init__()
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._reset_count = 0
        # Modes = "live" / "backtest" (default - historic data)
        self.env_mode = os.environ.get("ENV_MODE", "backtest").lower()
        
        # Asset Universe config
        universe_str = os.environ.get("ASSET_UNIVERSE", "AAPL,MSFT,TSLA,NVDA,SPY")
        self.asset_universe = [s.strip() for s in universe_str.split(",") if s.strip()]
        
        # Episode settings
        self.initial_virtual_cash = float(os.environ.get("INITIAL_VIRTUAL_CASH", "10000.0"))
        self.max_steps = int(os.environ.get("MAX_STEPS", "100"))
        
        # Virtual Ledger State
        self.virtual_cash = self.initial_virtual_cash
        self.virtual_positions: Dict[str, float] = {}
        
        # Reward shaping tracking
        self._consecutive_holds = 0
        
        # Initialize the broker interface: use Alpaca MCP for live trading,
        # or the SimulatedBroker for offline historical backtesting.
        if self.env_mode == "live":
            self.mcp_manager = MCPClientManager()
        else:
            self.mcp_manager = SimulatedBroker(self.asset_universe)

    def _get_market_prices(self) -> Dict[str, float]:
        """Fetch latest prices for the asset universe."""
        prices = {}
        if not getattr(self, "mcp_manager", None):
            return prices

        for asset in self.asset_universe:
            res = self.mcp_manager.call_tool("get_stock_latest_quote", {"symbols": asset, "feed": "iex"})
            if res.get("success") and res.get("result"):
                try:
                    text = res["result"][0].get("text", "")
                    data = json.loads(text)
                    quotes = data.get("quotes", {})
                    if asset in quotes:
                        quote_data = quotes[asset]
                        # Get AP(ask price) but if its missing fallback to BP(bid price)
                        if "ap" in quote_data and float(quote_data["ap"]) > 0:
                            prices[asset] = float(quote_data["ap"])
                        elif "bp" in quote_data and float(quote_data["bp"]) > 0:
                            prices[asset] = float(quote_data["bp"])
                except Exception as e:
                    pass
        return prices

    def _get_virtual_equity(self, market_prices: Dict[str, float]) -> float:
        """Calculate the current value of the virtual portfolio."""
        equity = self.virtual_cash
        for symbol, qty in self.virtual_positions.items():
            price = market_prices.get(symbol, 0.0)
            equity += qty * price
        return equity

    def _liquidate_all_virtual_positions(self):
        """Close out all positions held in the current episode."""
        if self.env_mode != "live" or not self.mcp_manager:
            self.virtual_positions.clear()
            self.virtual_cash = self.initial_virtual_cash
            return

        for symbol, qty in self.virtual_positions.items():
            if qty > 0:
                self.mcp_manager.call_tool(
                    "place_stock_order", 
                    {"symbol": symbol, "qty": qty, "side": "sell", "type": "market", "time_in_force": "ioc"}
                )
        self.virtual_positions.clear()
        self.virtual_cash = self.initial_virtual_cash

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> TradingObservation:
        """
        Reset the environment, liquidate previous holdings, and set up the next episode.

        Args:
            seed: Optional random seed for reproducibility
            episode_id: Optional custom episode identifier
            **kwargs: Additional reset parameters

        Returns:
            TradingObservation with initial virtual portfolio state
        """
        # Liquidate open positions from previous episode
        self._liquidate_all_virtual_positions()

        eid = episode_id if episode_id else str(uuid4())
        self._state = State(episode_id=eid, step_count=0)
        self._reset_count += 1
        self._consecutive_holds = 0
        
        if self.env_mode == "backtest" and hasattr(self.mcp_manager, "reset_time"):
            self.mcp_manager.reset_time(0)
        
        market_prices = self._get_market_prices()
        equity = self._get_virtual_equity(market_prices)

        account_state = VirtualAccountState(
            cash=self.virtual_cash,
            equity=equity,
            positions=self.virtual_positions.copy()
        )

        return TradingObservation(
            result=[{"text": f"Environment reset. Virtual cash injected: ${self.initial_virtual_cash:.2f}"}],
            error=None,
            market_prices=market_prices,
            account_state=account_state,
            done=False,
            reward=0.0,
        )

    def step(self, action: TradingAction) -> TradingObservation:  # type: ignore[override]
        """
        Executes a single step in the Reinforcement Learning environment.
        
        This method is the core engine of the environment. It performs:
        1. Action interception: Validates if the agent has enough virtual cash/positions to execute a trade.
        2. Virtual Accounting: Manages the 'paper' bounds so the agent can't cheat or over-leverage.
        3. Time-Stepping: Advances the simulator clock (if backtesting).
        4. Reward Calculation: Calculates the exact profit/loss delta caused by the action and market movement.
        """
        
        # 1. Capture the 'Before' State
        self._state.step_count += 1
        market_prices = self._get_market_prices()
        prev_equity = self._get_virtual_equity(market_prices)

        res = {"success": False, "result": [], "error": "Unknown error"}
        
        # 2. Intercept and Validate Trading Orders (The Virtual Ledger)
        if action.tool_name == "place_stock_order":
            symbol = action.tool_args.get("symbol")
            raw_qty = action.tool_args.get("qty", 0)
            qty = float(raw_qty)
            side = action.tool_args.get("side", "").lower()
            
            # Formatting Requirement: Alpaca API strongly expects 'qty' as a string
            action.tool_args["qty"] = str(raw_qty)
            
            # Fetch an estimated execution price to calculate if the agent can afford the trade
            est_price = market_prices.get(symbol)
            if est_price is None:
                # Need an estimate to allow the trade
                if self.mcp_manager:
                    q_res = self.mcp_manager.call_tool("get_stock_latest_quote", {"symbols": symbol, "feed": "iex"})
                    try:
                        data = json.loads(q_res["result"][0].get("text", ""))
                        quotes = data.get("quotes", {})
                        if symbol in quotes:
                            quote_data = quotes[symbol]
                            if "ap" in quote_data and float(quote_data["ap"]) > 0:
                                est_price = float(quote_data["ap"])
                            elif "bp" in quote_data and float(quote_data["bp"]) > 0:
                                est_price = float(quote_data["bp"])
                    except Exception:
                        pass
                # Virtual funds safety fallback
            if est_price is None: est_price = 100.0 
                
            est_cost = est_price * qty
            
            # 2a. Process BUY Orders
            if side == "buy":
                if est_cost > self.virtual_cash:
                    # BLOCK the trade: Agent relies on hallucinated funds
                    res["error"] = f"Insufficient Virtual Funds. Required: ${est_cost:.2f}, Available: ${self.virtual_cash:.2f}"
                else:
                    # ALLOW the trade: Deduct cash, add position, and forward to actual API 
                    self.virtual_cash -= est_cost
                    self.virtual_positions[symbol] = self.virtual_positions.get(symbol, 0) + qty
                    if self.mcp_manager:
                        res = self.mcp_manager.call_tool(action.tool_name, action.tool_args)
            
            # 2b. Process SELL Orders
            elif side == "sell":
                current_qty = self.virtual_positions.get(symbol, 0)
                if qty > current_qty:
                    # BLOCK the trade: Agent is trying to short or sell unowned shares
                    res["error"] = f"Insufficient Virtual Positions. Requested: {qty}, Available: {current_qty}"
                else:
                    # ALLOW the trade: Remove position, add cash back, and forward to API
                    self.virtual_positions[symbol] -= qty
                    self.virtual_cash += est_cost
                    if self.virtual_positions[symbol] <= 0:
                        del self.virtual_positions[symbol]
                    if self.mcp_manager:
                        res = self.mcp_manager.call_tool(action.tool_name, action.tool_args)
            else:
                 res["error"] = "Invalid order side, use 'buy' or 'sell'"
                 
        # 2c. Explicit Hold Action
        elif action.tool_name.lower() in ["hold", "pass"]:
            res = {"success": True, "result": [{"text": "Held positions. No trades placed."}]}
            
        else:
            # 2d. Non-Trading Tools (e.g. get_account_info) pass directly through
            if self.mcp_manager:
                res = self.mcp_manager.call_tool(action.tool_name, action.tool_args)
            else:
                res = {"success": True, "result": [{"text": f"Simulated call for {action.tool_name}"}]}

        # 3. Advance the Simulation Clock
        # This is where 'Day T' turns into 'Day T+1'. A backtested trade placed securely above
        # will now experience the future market price movement!
        if self.env_mode == "backtest" and hasattr(self.mcp_manager, "step_time"):
            self.mcp_manager.step_time()
            
        market_prices = self._get_market_prices()
        new_equity = self._get_virtual_equity(market_prices)
        
        # 4. Calculate RL Reward Signal
        # Base reward: equity delta (how much money the agent's actions + market created)
        reward = new_equity - prev_equity
        
        # ── Reward Shaping: Penalties & Bonuses ──
        action_was_rejected = not res.get("success", False) and action.tool_name == "place_stock_order"
        is_hold = action.tool_name.lower() in ["hold", "pass"]
        
        # Penalty: Invalid/rejected trade attempts (agent tried to trade with insufficient funds/positions)
        if action_was_rejected:
            reward -= 10.0  # $10 penalty for invalid orders
        
        # Penalty: Consecutive holds with no positions (discourages do-nothing agents)
        if is_hold and len(self.virtual_positions) == 0:
            self._consecutive_holds += 1
            if self._consecutive_holds > 3:
                reward -= 5.0 * (self._consecutive_holds - 3)  # Escalating penalty
        else:
            self._consecutive_holds = 0
        
        # Bonus: Diversification (holding >= 2 different assets)
        if len(self.virtual_positions) >= 2:
            reward += 2.0  # Small bonus for portfolio diversification
        
        # 5. Check Termination (Episode Done) Conditions
        done = False
        if self._state.step_count >= self.max_steps:
            done = True # Max steps reached
        # [IMPORTANT Param]
        elif new_equity < (self.initial_virtual_cash * 0.8): 
            done = True # Margin call! Agent lost 80% of the account

        # 6. Formulate the returned Observation State for the LLM prompt
        account_state = VirtualAccountState(
            cash=self.virtual_cash,
            equity=new_equity,
            positions=self.virtual_positions.copy()
        )

        return TradingObservation(
            result=res.get("result", []),
            error=res.get("error", None) if not res.get("success") else None,
            market_prices=market_prices,
            account_state=account_state,
            done=done,
            reward=reward,
            metadata={
                "tool_called": action.tool_name, 
                "step": self._state.step_count,
                "reward_delta": reward,
                "reward_shaping": {
                    "equity_delta": new_equity - prev_equity,
                    "invalid_trade_penalty": -10.0 if action_was_rejected else 0.0,
                    "hold_penalty": -5.0 * max(0, self._consecutive_holds - 3) if is_hold and len(self.virtual_positions) == 0 else 0.0,
                    "diversification_bonus": 2.0 if len(self.virtual_positions) >= 2 else 0.0,
                },
            },
        )
    
    @property
    def state(self) -> State:
        """Get the current environment state."""
        return self._state
