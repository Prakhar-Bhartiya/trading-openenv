"""
Baseline Inference Script for the Trading OpenEnv Environment.

This script uses the OpenAI API client to run a model against the
environment across all 3 defined tasks. It reads API credentials from
the OPENAI_API_KEY environment variable and produces reproducible
baseline scores.

Usage:
    export OPENAI_API_KEY="sk-..."
    python baseline_agent.py

    # Or with a specific model:
    export OPENAI_MODEL="gpt-5.4-mini"
    python baseline_agent.py
"""

import os
import sys
import json
import re
import time
from typing import List, Dict, Any

from openai import OpenAI

# Ensure local imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from trading.models import TradingAction
from trading.server.trading_environment import TradingEnvironment
from trading.tasks import TASKS, grade_trajectory
from trading.tasks.task_definitions import TASK_ORDER


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("❌ Error: OPENAI_API_KEY environment variable is not set.")
    print("   Set it with: export OPENAI_API_KEY='sk-...'")
    sys.exit(1)

MODEL = os.environ.get("OPENAI_MODEL", "gpt-5.4-mini")
client = OpenAI(api_key=OPENAI_API_KEY)


# Keep only the last N user/assistant exchanges (plus system prompt) to avoid
# context-window blowup and rate-limit errors on longer episodes.
MESSAGE_WINDOW = 12  # number of recent messages to retain

SYSTEM_PROMPT = """You are an autonomous RL trading AI playing an episodic trading game.
You start each episode with $10,000 virtual cash. Your goal is to maximize equity by actively trading stocks.

At each turn you receive the current market state including prices, your account (cash, equity, positions), and the result of your last action.

IMPORTANT STRATEGY GUIDELINES:
- Actively look for buying opportunities when you hold mostly cash.
- Don't just hold cash for long stretches — look at price trends and re-enter positions.
- Diversify across the available assets when possible.
- Use position sizing to manage risk (don't go all-in on one stock).

Respond with brief reasoning, then exactly ONE action block in <action> tags containing valid JSON.

To trade:
<action>
{
    "tool_name": "place_stock_order",
    "tool_args": {
        "symbol": "AAPL",
        "qty": "5",
        "side": "buy",
        "type": "market",
        "time_in_force": "ioc"
    }
}
</action>

To hold (do nothing this step):
<action>
{
    "tool_name": "hold",
    "tool_args": {}
}
</action>
"""


# ─────────────────────────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────────────────────────

def extract_action(text: str) -> dict:
    """Extracts the JSON from the <action> tags."""
    match = re.search(r'<action>(.*?)</action>', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            return None
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Run a Single Task
# ─────────────────────────────────────────────────────────────────────────────

def run_task(task_id: str) -> tuple[float, List[Dict[str, Any]]]:
    """Run a single task and return (score, trajectory).

    Returns:
        Tuple of (graded_score, trajectory_log)
    """
    task = TASKS[task_id]
    print(f"\n{'─' * 60}")
    print(f"  📋 Task: {task.name} ({task.difficulty})")
    print(f"  📝 {task.description}")
    print(f"  ⏱️  Max Steps: {task.max_steps}")
    print(f"{'─' * 60}")

    # Apply task-level environment overrides
    env_overrides = task.to_env_overrides()
    for k, v in env_overrides.items():
        os.environ[k] = v

    # Create and reset environment
    env = TradingEnvironment()
    obs = env.reset(seed=task.seed, episode_id=f"baseline-{task_id}")
    initial_equity = obs.account_state.equity

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    system_msg = messages[0]
    trajectory: List[Dict[str, Any]] = []

    step = 0
    while not obs.done:
        step += 1

        # Build observation prompt
        obs_text = json.dumps({
            "step": env.state.step_count,
            "market_prices": obs.market_prices,
            "account_state": obs.account_state.model_dump(),
            "last_action_result": obs.result,
            "last_action_error": obs.error,
            "step_reward": obs.reward,
        }, indent=2)

        messages.append({
            "role": "user",
            "content": f"Current State:\n{obs_text}\n\nWhat is your next action?",
        })

        # Trim context to sliding window (system + last N messages)
        if len(messages) > MESSAGE_WINDOW + 1:
            messages = [system_msg] + messages[-(MESSAGE_WINDOW):]

        # Call OpenAI API
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=0.2,  # Low temp for reproducibility
            )
            llm_reply = response.choices[0].message.content
            messages.append({"role": "assistant", "content": llm_reply})
        except Exception as e:
            print(f"  ❌ OpenAI API Error at step {step}: {e}")
            break

        # Parse action
        action_dict = extract_action(llm_reply)
        if not action_dict:
            action_dict = {"tool_name": "hold", "tool_args": {}}

        action = TradingAction(
            tool_name=action_dict.get("tool_name", "hold"),
            tool_args=action_dict.get("tool_args", {}),
        )

        # Execute step
        obs = env.step(action)

        # Log trajectory
        trajectory.append({
            "step": env.state.step_count,
            "equity": obs.account_state.equity,
            "reward": obs.reward,
            "action": action_dict,
        })

        # Progress indicator
        equity_str = f"${obs.account_state.equity:,.2f}"
        action_str = action_dict.get("tool_name", "?")
        if action_str == "place_stock_order":
            args = action_dict.get("tool_args", {})
            action_str = f"{args.get('side', '?').upper()} {args.get('qty', '?')} {args.get('symbol', '?')}"
        elif action_str in ("hold", "pass"):
            action_str = "HOLD"

        print(f"  Step {step:3d}/{task.max_steps} | Equity: {equity_str:>12s} | Action: {action_str}")

        time.sleep(0.5)  # Rate-limit protection

    # Grade trajectory
    final_equity = obs.account_state.equity
    score = grade_trajectory(
        task_id=task_id,
        trajectory=trajectory,
        initial_equity=initial_equity,
        final_equity=final_equity,
    )

    profit = final_equity - initial_equity
    print(f"\n  📊 Results:")
    print(f"     Initial Equity: ${initial_equity:,.2f}")
    print(f"     Final Equity:   ${final_equity:,.2f}")
    print(f"     P/L:            ${profit:+,.2f}")
    print(f"     🏆 Score:        {score:.3f} / 1.000")

    return score, trajectory


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  🤖 Trading OpenEnv — Baseline Inference")
    print(f"  Model: {MODEL}")
    print(f"  Tasks: {len(TASK_ORDER)}")
    print("=" * 60)

    results: Dict[str, float] = {}

    for task_id in TASK_ORDER:
        try:
            score, _ = run_task(task_id)
            results[task_id] = score
        except Exception as e:
            print(f"\n  ❌ Task '{task_id}' failed with error: {e}")
            results[task_id] = 0.0

    # Final Summary
    print("\n" + "=" * 60)
    print("  📋 BASELINE RESULTS SUMMARY")
    print("=" * 60)
    print(f"  {'Task':<30s} {'Difficulty':<10s} {'Score':>10s}")
    print(f"  {'─' * 50}")

    total = 0.0
    for task_id in TASK_ORDER:
        task = TASKS[task_id]
        score = results.get(task_id, 0.0)
        total += score
        emoji = "✅" if score >= 0.5 else "⚠️" if score > 0 else "❌"
        print(f"  {emoji} {task.name:<28s} {task.difficulty:<10s} {score:>8.3f}")

    avg = total / len(TASK_ORDER) if TASK_ORDER else 0.0
    print(f"  {'─' * 50}")
    print(f"  {'Average Score':<40s} {avg:>8.3f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
