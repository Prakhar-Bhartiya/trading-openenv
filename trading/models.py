# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Trading Environment.

The trading environment wraps the Alpaca MCP server.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from openenv.core.env_server.types import Action, Observation


class TradingAction(Action):
    """Action for the Trading environment - specifying an MCP tool to call."""

    tool_name: str = Field(..., description="Name of the MCP tool to execute")
    tool_args: Dict[str, Any] = Field(default_factory=dict, description="Arguments for the tool")


class VirtualAccountState(BaseModel):
    """Virtual account state tracking only the current episode's progress."""
    cash: float = Field(default=10000.0, description="Virtual cash remaining in this episode")
    equity: float = Field(default=10000.0, description="Virtual equity (cash + position value)")
    positions: Dict[str, float] = Field(default_factory=dict, description="Assets and quantities held in this episode")


class TradingObservation(Observation):
    """Observation from the Trading environment - result of the tool execution."""

    result: List[Dict[str, Any]] = Field(default_factory=list, description="The tool output/content")
    error: Optional[str] = Field(default=None, description="Error message if the tool failed")
    market_prices: Dict[str, float] = Field(default_factory=dict, description="Latest prices for the asset universe")
    account_state: VirtualAccountState = Field(default_factory=VirtualAccountState, description="Virtual ledger state")
