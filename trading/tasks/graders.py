"""
Programmatic Graders for the Trading OpenEnv Environment.

Each grader takes a trajectory (list of step records) and returns a
deterministic score in the range [0.0, 1.0].

Graders are designed to:
  - Be fully deterministic (no randomness, no LLM calls)
  - Reward partial progress (not just binary pass/fail)
  - Have clear, documented success/failure criteria
"""

from typing import Dict, Any, List
from .task_definitions import TaskConfig, TASKS


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp a value to [lo, hi]."""
    return max(lo, min(hi, value))


def _compute_max_drawdown(equity_curve: List[float]) -> float:
    """Compute maximum drawdown from an equity curve.

    Returns a value in [0, 1] where 0 = no drawdown and 1 = total loss.
    """
    if not equity_curve:
        return 0.0
    peak = equity_curve[0]
    max_dd = 0.0
    for eq in equity_curve:
        if eq > peak:
            peak = eq
        dd = (peak - eq) / peak if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd
    return max_dd


def _count_trades(trajectory: List[Dict[str, Any]]) -> int:
    """Count the number of actual trade actions (buy/sell) in a trajectory."""
    count = 0
    for step in trajectory:
        action = step.get("action", {})
        tool = action.get("tool_name", "")
        if tool == "place_stock_order":
            count += 1
    return count


# ─────────────────────────────────────────────────────────────────────────────
# Grader: Capital Preservation (Easy)
# ─────────────────────────────────────────────────────────────────────────────

def grade_capital_preservation(
    trajectory: List[Dict[str, Any]],
    initial_equity: float,
    final_equity: float,
    params: Dict[str, Any],
) -> float:
    """Grade the Capital Preservation task.

    Scoring:
      - final_equity >= perfect_threshold → 1.0
      - final_equity <= zero_threshold    → 0.0
      - Linear interpolation between the two thresholds

    Args:
        trajectory: List of step records
        initial_equity: Starting equity ($10,000)
        final_equity: Equity at end of episode
        params: Grader parameters from TaskConfig

    Returns:
        Score in [0.0, 1.0]
    """
    perfect = params.get("perfect_threshold", 9_500.0)
    zero = params.get("zero_threshold", 8_000.0)

    if final_equity >= perfect:
        return 1.0
    if final_equity <= zero:
        return 0.0

    # Linear interpolation
    return _clamp((final_equity - zero) / (perfect - zero))


# ─────────────────────────────────────────────────────────────────────────────
# Grader: Profitable Episode (Medium)
# ─────────────────────────────────────────────────────────────────────────────

def grade_profitable_episode(
    trajectory: List[Dict[str, Any]],
    initial_equity: float,
    final_equity: float,
    params: Dict[str, Any],
) -> float:
    """Grade the Profitable Episode task.

    Scoring:
      - Base score: linear scale from $0 profit (0.0) to $500 profit (1.0)
      - Negative profit → 0.0
      - A small bonus (up to 0.1) for meeting minimum trade activity

    Args:
        trajectory: List of step records
        initial_equity: Starting equity ($10,000)
        final_equity: Equity at end of episode
        params: Grader parameters from TaskConfig

    Returns:
        Score in [0.0, 1.0]
    """
    profit = final_equity - initial_equity
    profit_for_perfect = params.get("profit_for_perfect", 500.0)
    min_trades = params.get("min_trades_for_bonus", 5)

    # Base profit score (90% weight)
    if profit <= 0:
        profit_score = 0.0
    else:
        profit_score = _clamp(profit / profit_for_perfect)

    # Activity bonus (10% weight) — must actually trade
    trade_count = _count_trades(trajectory)
    activity_score = _clamp(trade_count / min_trades)

    return _clamp(0.9 * profit_score + 0.1 * activity_score)


# ─────────────────────────────────────────────────────────────────────────────
# Grader: Drawdown-Controlled Alpha (Hard)
# ─────────────────────────────────────────────────────────────────────────────

def grade_drawdown_controlled_alpha(
    trajectory: List[Dict[str, Any]],
    initial_equity: float,
    final_equity: float,
    params: Dict[str, Any],
) -> float:
    """Grade the Drawdown-Controlled Alpha task.

    Composite scoring:
      - Profit component (50%): Linear from $0 to $1,000
      - Drawdown component (30%): 1.0 if max_dd < 10%, decays to 0.0 at 30%
      - Activity component (20%): Linear scaling to minimum 10 trades

    Args:
        trajectory: List of step records
        initial_equity: Starting equity ($10,000)
        final_equity: Equity at end of episode
        params: Grader parameters from TaskConfig

    Returns:
        Score in [0.0, 1.0]
    """
    w_profit = params.get("weight_profit", 0.50)
    w_drawdown = params.get("weight_drawdown", 0.30)
    w_activity = params.get("weight_activity", 0.20)

    # ── 1. Profit Component ──
    profit = final_equity - initial_equity
    profit_for_perfect = params.get("profit_for_perfect", 1_000.0)
    profit_score = _clamp(profit / profit_for_perfect) if profit > 0 else 0.0

    # ── 2. Drawdown Component ──
    equity_curve = [initial_equity] + [step.get("equity", initial_equity) for step in trajectory]
    max_dd = _compute_max_drawdown(equity_curve)

    dd_threshold = params.get("max_drawdown_threshold", 0.10)
    dd_zero = params.get("drawdown_zero_at", 0.30)

    if max_dd <= dd_threshold:
        drawdown_score = 1.0
    elif max_dd >= dd_zero:
        drawdown_score = 0.0
    else:
        drawdown_score = _clamp(1.0 - (max_dd - dd_threshold) / (dd_zero - dd_threshold))

    # ── 3. Activity Component ──
    trade_count = _count_trades(trajectory)
    min_trades = params.get("min_trades", 10)
    activity_score = _clamp(trade_count / min_trades)

    # ── Composite Score ──
    return _clamp(
        w_profit * profit_score +
        w_drawdown * drawdown_score +
        w_activity * activity_score
    )


# ─────────────────────────────────────────────────────────────────────────────
# Dispatcher
# ─────────────────────────────────────────────────────────────────────────────

_GRADER_REGISTRY = {
    "capital_preservation": grade_capital_preservation,
    "profitable_episode": grade_profitable_episode,
    "drawdown_controlled_alpha": grade_drawdown_controlled_alpha,
}


def grade_trajectory(
    task_id: str,
    trajectory: List[Dict[str, Any]],
    initial_equity: float,
    final_equity: float,
) -> float:
    """Grade a full trajectory for a given task.

    This is the main entry point for grading. It dispatches to the
    appropriate task-specific grader.

    Args:
        task_id: One of "capital_preservation", "profitable_episode",
                 "drawdown_controlled_alpha"
        trajectory: List of step dicts, each containing at minimum:
                    {"step": int, "equity": float, "reward": float, "action": dict}
        initial_equity: The starting equity ($10,000 by default)
        final_equity: The final equity at end of episode

    Returns:
        Score in [0.0, 1.0]

    Raises:
        KeyError: If task_id is unknown
    """
    if task_id not in _GRADER_REGISTRY:
        valid = ", ".join(_GRADER_REGISTRY.keys())
        raise KeyError(f"Unknown task '{task_id}'. Valid tasks: {valid}")

    task_config = TASKS[task_id]
    grader_fn = _GRADER_REGISTRY[task_id]

    return grader_fn(
        trajectory=trajectory,
        initial_equity=initial_equity,
        final_equity=final_equity,
        params=task_config.grader_params,
    )
