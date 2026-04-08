# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Trading Environment Client."""

from typing import Dict

from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State
from openenv.core import EnvClient

from .models import TradingAction, TradingObservation, VirtualAccountState


class TradingEnv(
    EnvClient[TradingAction, TradingObservation, State]
):
    """
    Client for the Trading Environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    """

    def _step_payload(self, action: TradingAction) -> Dict:
        """
        Convert TradingAction to JSON payload for step message.

        Args:
            action: TradingAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        return {
            "tool_name": action.tool_name,
            "tool_args": action.tool_args,
        }

    def _parse_result(self, payload: Dict) -> StepResult[TradingObservation]:
        """
        Parse server response into StepResult[TradingObservation].

        Args:
            payload: JSON response data from server

        Returns:
            StepResult with TradingObservation
        """
        obs_data = payload.get("observation", {})

        # Parse the nested account_state
        acct_data = obs_data.get("account_state", {})
        account_state = VirtualAccountState(
            cash=acct_data.get("cash", 10000.0),
            equity=acct_data.get("equity", 10000.0),
            positions=acct_data.get("positions", {}),
        )

        observation = TradingObservation(
            result=obs_data.get("result", []),
            error=obs_data.get("error", None),
            market_prices=obs_data.get("market_prices", {}),
            account_state=account_state,
            done=payload.get("done", False),
            reward=payload.get("reward", 0.0),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """
        Parse server response into State object.

        Args:
            payload: JSON response from state request

        Returns:
            State object with episode_id and step_count
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
