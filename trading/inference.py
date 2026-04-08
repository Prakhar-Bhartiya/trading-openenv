"""
Inference Script — Trading OpenEnv
"""

import os
import sys
import json
import re
import time
from typing import List, Dict, Any, Optional

from openai import OpenAI

# Ensure local imports work (both installed package and direct execution)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from trading.models import TradingAction
from trading.server.trading_environment import TradingEnvironment
from trading.tasks import TASKS, grade_trajectory
from trading.tasks.task_definitions import TASK_ORDER


# ─────────────────────────────────────────────────────────────────────────────
# Mandatory Environment Variables
# ─────────────────────────────────────────────────────────────────────────────

IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")


BENCHMARK = "trading"
TEMPERATURE = 0.2
MESSAGE_WINDOW = 12  # Rolling context window size (system + last N messages)


# ─────────────────────────────────────────────────────────────────────────────
# System Prompt
# ─────────────────────────────────────────────────────────────────────────────

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

def log_start(task: str, env: str, model: str) -> None:
    """Emit the mandatory [START] line."""
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    """Emit the mandatory [STEP] line."""
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    """Emit the mandatory [END] line."""
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────────────────────────

def extract_action(text: str) -> Optional[dict]:
    """Extract the JSON action dictionary from <action> tags in LLM response."""
    match = re.search(r'<action>(.*?)</action>', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            return None
    return None


def format_action_str(action_dict: dict) -> str:
    """Format an action dict into a compact string for [STEP] logging.

    Examples:
        place_stock_order(AAPL,5,buy)
        hold()
    """
    tool = action_dict.get("tool_name", "hold")
    if tool == "place_stock_order":
        args = action_dict.get("tool_args", {})
        sym = args.get("symbol", "?")
        qty = args.get("qty", "?")
        side = args.get("side", "?")
        return f"place_stock_order({sym},{qty},{side})"
    return f"{tool}()"


def get_model_response(
    client: OpenAI,
    messages: List[dict],
) -> str:
    """Call the OpenAI-compatible LLM and return the response text."""
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=TEMPERATURE,
            stream=False,
        )
        text = (response.choices[0].message.content or "").strip()
        return text if text else '<action>{"tool_name":"hold","tool_args":{}}</action>'
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return '<action>{"tool_name":"hold","tool_args":{}}</action>'


# ─────────────────────────────────────────────────────────────────────────────
# Run a Single Task
# ─────────────────────────────────────────────────────────────────────────────

def run_task(task_id: str, client: OpenAI) -> float:
    """Run a single task, emitting [START]/[STEP]/[END] lines.

    Returns:
        Graded score in [0.0, 1.0]
    """
    task = TASKS[task_id]

    # Apply task-level environment overrides
    env_overrides = task.to_env_overrides()
    for k, v in env_overrides.items():
        os.environ[k] = v

    # Create and reset environment
    env = TradingEnvironment()
    obs = env.reset(seed=task.seed, episode_id=f"inference-{task_id}")
    initial_equity = obs.account_state.equity

    messages: List[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    system_msg = messages[0]
    trajectory: List[Dict[str, Any]] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    # ── [START] ──
    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        for step in range(1, task.max_steps + 1):
            if obs.done:
                break

            # Build observation prompt for the LLM
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

            # Call LLM via OpenAI client
            llm_reply = get_model_response(client, messages)
            messages.append({"role": "assistant", "content": llm_reply})

            # Parse action from LLM response
            action_dict = extract_action(llm_reply)
            if not action_dict:
                action_dict = {"tool_name": "hold", "tool_args": {}}

            action = TradingAction(
                tool_name=action_dict.get("tool_name", "hold"),
                tool_args=action_dict.get("tool_args", {}),
            )

            # Execute step in environment
            obs = env.step(action)

            reward = obs.reward or 0.0
            rewards.append(reward)
            steps_taken = step

            # Track trajectory for grading
            trajectory.append({
                "step": env.state.step_count,
                "equity": obs.account_state.equity,
                "reward": reward,
                "action": action_dict,
            })

            # ── [STEP] ──
            action_str = format_action_str(action_dict)
            log_step(
                step=step,
                action=action_str,
                reward=reward,
                done=obs.done,
                error=obs.error,
            )

            if obs.done:
                break

            time.sleep(0.3)  # Rate-limit protection

        # Grade the completed trajectory
        final_equity = obs.account_state.equity
        score = grade_trajectory(
            task_id=task_id,
            trajectory=trajectory,
            initial_equity=initial_equity,
            final_equity=final_equity,
        )
        score = min(max(score, 0.0), 1.0)  # Clamp to [0, 1]
        success = score >= 0.1

    except Exception as e:
        print(f"[DEBUG] Task '{task_id}' error: {e}", flush=True)

    finally:
        # ── [END] ── (always emitted, even on exception)
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    if not API_KEY:
        print(
            "[DEBUG] Warning: HF_TOKEN or API_KEY not set. LLM calls will fail.",
            flush=True,
        )

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    for task_id in TASK_ORDER:
        try:
            run_task(task_id, client)
        except Exception as e:
            print(f"[DEBUG] Task '{task_id}' crashed: {e}", flush=True)
            # Always emit [END] even on crash
            log_end(success=False, steps=0, score=0.0, rewards=[])


if __name__ == "__main__":
    main()
