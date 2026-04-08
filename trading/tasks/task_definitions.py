"""
Task Definitions for the Trading OpenEnv Environment.

Each task defines:
  - A concrete objective an agent must accomplish
  - Configuration (max_steps, initial_cash, difficulty)
  - Environment overrides applied at reset time
  - Success criteria used by the corresponding grader

Tasks range from Easy → Medium → Hard.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional


@dataclass
class TaskConfig:
    """Configuration for a single evaluation task."""

    task_id: str
    name: str
    difficulty: str  # "easy", "medium", "hard"
    description: str
    max_steps: int
    initial_cash: float = 10_000.0
    asset_universe: list[str] = field(default_factory=lambda: ["AAPL", "MSFT", "TSLA", "NVDA", "SPY"])
    seed: Optional[int] = None  # For reproducibility

    # Grader-specific thresholds (used by graders.py)
    grader_params: Dict[str, Any] = field(default_factory=dict)

    def to_env_overrides(self) -> Dict[str, str]:
        """Return environment variable overrides for this task."""
        return {
            "MAX_STEPS": str(self.max_steps),
            "INITIAL_VIRTUAL_CASH": str(self.initial_cash),
            "ASSET_UNIVERSE": ",".join(self.asset_universe),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Task 1: Capital Preservation (Easy)
# ─────────────────────────────────────────────────────────────────────────────
TASK_CAPITAL_PRESERVATION = TaskConfig(
    task_id="capital_preservation",
    name="Capital Preservation",
    difficulty="easy",
    description=(
        "Preserve your capital over a short trading window. "
        "Start with $10,000 and avoid losing more than 5% over 20 steps. "
        "An agent that simply holds and does nothing should score well. "
        "The grader rewards ending above $9,500 and penalizes large losses."
    ),
    max_steps=20,
    initial_cash=10_000.0,
    seed=42,
    grader_params={
        "perfect_threshold": 9_500.0,   # >= this → score 1.0
        "zero_threshold": 8_000.0,      # <= this → score 0.0
    },
)

# ─────────────────────────────────────────────────────────────────────────────
# Task 2: Profitable Episode (Medium)
# ─────────────────────────────────────────────────────────────────────────────
TASK_PROFITABLE_EPISODE = TaskConfig(
    task_id="profitable_episode",
    name="Profitable Episode",
    difficulty="medium",
    description=(
        "Generate positive returns over a 50-step episode. "
        "Start with $10,000 and try to grow your portfolio. "
        "The grader rewards profit on a linear scale: $0 profit = 0.0, "
        "$500+ profit = 1.0. You must actively trade to score well."
    ),
    max_steps=50,
    initial_cash=10_000.0,
    seed=42,
    grader_params={
        "profit_for_perfect": 500.0,    # $500 profit → 1.0
        "min_trades_for_bonus": 5,      # Activity threshold
    },
)

# ─────────────────────────────────────────────────────────────────────────────
# Task 3: Drawdown-Controlled Alpha (Hard)
# ─────────────────────────────────────────────────────────────────────────────
TASK_DRAWDOWN_CONTROLLED_ALPHA = TaskConfig(
    task_id="drawdown_controlled_alpha",
    name="Drawdown-Controlled Alpha",
    difficulty="hard",
    description=(
        "Maximize returns over 100 steps while maintaining strict risk control. "
        "Keep maximum drawdown under 10% while generating positive alpha. "
        "Grading is a weighted composite: profit (50%), drawdown control (30%), "
        "and trading activity (20%). This requires balancing aggression with "
        "risk management — simply holding won't score well on the activity component."
    ),
    max_steps=100,
    initial_cash=10_000.0,
    seed=42,
    grader_params={
        "profit_for_perfect": 1_000.0,  # $1,000 profit → full profit score
        "max_drawdown_threshold": 0.10, # 10% max drawdown for full dd score
        "drawdown_zero_at": 0.30,       # 30% drawdown → 0 dd score
        "min_trades": 10,               # Need >=10 trades for full activity score
        "weight_profit": 0.50,
        "weight_drawdown": 0.30,
        "weight_activity": 0.20,
    },
)


# ─────────────────────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────────────────────
TASKS: Dict[str, TaskConfig] = {
    "capital_preservation": TASK_CAPITAL_PRESERVATION,
    "profitable_episode": TASK_PROFITABLE_EPISODE,
    "drawdown_controlled_alpha": TASK_DRAWDOWN_CONTROLLED_ALPHA,
}

TASK_ORDER = ["capital_preservation", "profitable_episode", "drawdown_controlled_alpha"]


def get_task(task_id: str) -> TaskConfig:
    """Get a task by its ID. Raises KeyError if not found."""
    if task_id not in TASKS:
        valid = ", ".join(TASKS.keys())
        raise KeyError(f"Unknown task '{task_id}'. Valid tasks: {valid}")
    return TASKS[task_id]
