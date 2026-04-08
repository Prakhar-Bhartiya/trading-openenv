"""
Simulator Engine Module

This module contains the historical data backtesting engine for the Trading Environment.
"""
import json
import logging
from typing import Dict, Any, List
from datetime import datetime

import pandas as pd
import yfinance as yf

class SimulatedBroker:
    """
    Simulates a broker using historical data. Mocking Alpaca MCP interface.
    """
    
    def __init__(self, asset_universe: List[str]):
        self.asset_universe = asset_universe
        self.data_store: Dict[str, pd.DataFrame] = {}
        self.current_step = 0
        self.max_steps = 0
        
        self.load_data()
        
    def load_data(self):
        """Loads historical data for the asset universe using yfinance."""
        print(f"Loading historical data for {self.asset_universe} using yfinance (1y daily)...")
        for sym in self.asset_universe:
            try:
                # yf.Ticker(sym).history(period="1y") gets 1 yr of daily bars
                df = yf.Ticker(sym).history(period="1y")
                if not df.empty:
                    self.data_store[sym] = df
                else:
                    print(f"Warning: No data fetched for {sym}")
            except Exception as e:
                print(f"Error fetching data for {sym}: {e}")
                
        if self.data_store:
            # Find the minimum max_steps across all assets to prevent out of bounds
            lengths = [len(df) for df in self.data_store.values()]
            self.max_steps = min(lengths) - 1
            print(f"✅ Loaded {len(self.data_store)} assets. Capable of {self.max_steps} daily steps from {self.data_store[self.asset_universe[0]].index[0].date()} to {self.data_store[self.asset_universe[0]].index[-1].date()}")
        else:
            raise ValueError("Failed to load any historical data.")

    def reset_time(self, start_step: int = 0):
        """Resets the simulator to a specific step."""
        self.current_step = min(start_step, self.max_steps)

    def step_time(self):
        """Advances the internal clock by one timestep."""
        if self.current_step < self.max_steps:
            self.current_step += 1

    def call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Polymorphic implementation of the MCP client's call_tool."""
        if name == "get_stock_latest_quote":
            # Match recent fix: using 'symbols'
            symbol = arguments.get("symbols", "") 
            if not symbol or symbol not in self.data_store:
                return {"success": False, "error": f"Symbol {symbol} not in loaded universe."}
                
            df = self.data_store[symbol]
            idx = min(self.current_step, len(df) - 1)
            row = df.iloc[idx]
            
            try:
                price = float(row["Close"])
            except Exception:
                price = 100.0 # Extreme fallback
                
            try:
                date_str = str(df.index[idx].date())
            except Exception:
                date_str = "2026-01-01"

            # Spoof the exact Alpaca format our environment parses!
            # Format: {"quotes":{"AAPL":{"ap":266.91}}}
            spoof_json = {
                "quotes": {
                    symbol: {
                        "ap": price,
                        "bp": price,
                        "t": date_str
                    }
                }
            }
            return {
                "success": True, 
                "result": [{"type": "text", "text": json.dumps(spoof_json)}]
            }
            
        elif name == "place_stock_order":
            symbol = arguments.get("symbol")
            qty = arguments.get("qty")
            side = arguments.get("side")
            
            # Since the environment's VirtualLedger tracks exactly whether an order is allowed
            # we just need to return a success to ensure the environment knows we accepted it.
            return {
                "success": True, 
                "result": [{"type": "text", "text": f"Simulated order placed. {side} {qty} {symbol}."}]
            }
            
        else:
            return {"success": True, "result": [{"type": "text", "text": f"Simulated call mocked for {name}."}]}
