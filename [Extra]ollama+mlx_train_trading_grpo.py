"""
GRPO Trading Training Pipeline — Local Ollama + MLX

Train a local LLM to discover profitable trading patterns using
Group Relative Policy Optimization (GRPO) via reinforcement learning.

Architecture:
  - Ollama (lfm2.5-thinking) → fast rollout generation (πθ_old)
  - MLX (LFM-2.5-1.2B-Thinking-MLX-5bit) → trainable policy (πθ) + frozen reference (π_ref)
  - TradingEnvironment (backtest mode) → SimulatedBroker with yfinance historical data

Usage:
  python train_trading_grpo.py
  python train_trading_grpo.py --max-train-steps 10 --num-episodes 2  # quick smoke test
"""

import argparse
import copy
import json
import math
import os
import random
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

import psutil

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TradingGRPOConfig:
    """Training configuration for GRPO on Apple Silicon."""

    output_dir: str = "trading-grpo-lfm2.5"
    run_name: str = "lfm2.5-trading-grpo"

    # ── Environment ──
    max_env_steps: int = 100           # Steps per trading episode
    asset_universe: str = "AAPL,MSFT,TSLA,NVDA,SPY"
    initial_cash: float = 10_000.0

    # ── GRPO Training ──
    learning_rate: float = 3e-6
    num_episodes_per_step: int = 4     # Group size G — episodes per training step
    max_train_steps: int = 30          # Total GRPO training steps
    gradient_accumulation_steps: int = 2
    max_grad_norm: float = 0.1
    warmup_ratio: float = 0.1
    clip_eps: float = 0.2              # PPO clipping epsilon
    kl_coeff: float = 0.05             # KL penalty coefficient (β)
    turns_to_sample: int = 5           # Num representative turns per episode for GRPO

    # ── Ollama (rollout generation) ──
    ollama_model: str = "lfm2.5-thinking"
    temperature: float = 0.7

    # ── MLX (training) ──
    mlx_model: str = "LiquidAI/LFM2.5-1.2B-Thinking-MLX-5bit"

    # ── Checkpointing & Logging ──
    save_steps: int = 10
    logging_steps: int = 1

    seed: int = 42


# ─────────────────────────────────────────────────────────────────────────────
# System Prompt
# ─────────────────────────────────────────────────────────────────────────────

TRADING_SYSTEM_PROMPT = """\
You are an autonomous RL trading AI playing an episodic trading game.
You start each episode with a $10,000 virtual cash allowance. Your goal is to maximize your virtual equity by buying and selling stocks.
Your RL environment operates in discrete steps. At each turn, you will receive the current state:
- Market Prices for the tracked asset universe
- Your Virtual Account tracking (Remaining Cash, Total Equity, Open Positions)
- The raw result of your previous command.

You must respond with your thinking, followed by exactly ONE action block enclosed in <action> tags containing valid JSON.

Valid action to trade:
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

If you want to hold your positions and do nothing for this step, output:
<action>
{
    "tool_name": "hold",
    "tool_args": {}
}
</action>

STRATEGY GUIDELINES:
- Diversify across assets when possible
- Consider position sizing relative to your total equity
- Sell positions that are declining to cut losses
- Buy assets showing upward momentum
- Monitor your cash reserves; don't go all-in on one asset
- Think carefully about your reasoning before acting
"""


# ─────────────────────────────────────────────────────────────────────────────
# Utility: Action Extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_action(text: str) -> Optional[dict]:
    """Extracts the JSON from the <action> tags."""
    match = re.search(r'<action>(.*?)</action>', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            return None
    return None


def format_action_label(action_dict: dict) -> str:
    """Creates a human-readable action label."""
    name = action_dict.get("tool_name", "unknown")
    args = action_dict.get("tool_args", {})
    if name == "place_stock_order":
        return f"{args.get('side', '?').upper()} {args.get('qty', '?')} {args.get('symbol', '?')}"
    elif name in ("hold", "pass"):
        return "HOLD"
    return name


# ─────────────────────────────────────────────────────────────────────────────
# Rollout: Play One Trading Episode via Ollama
# ─────────────────────────────────────────────────────────────────────────────

def play_trading_episode(
    env,
    ollama_model: str,
    system_prompt: str,
    max_steps: int,
    temperature: float = 0.7,
) -> dict:
    """Play one full trading episode using Ollama for generation.

    Args:
        env: TradingEnvironment instance (reset externally before calling)
        ollama_model: Ollama model name for inference
        system_prompt: System prompt for the trading agent
        max_steps: Maximum environment steps
        temperature: Sampling temperature

    Returns:
        dict with per-step prompts, completions, rewards, equity, and episode metrics
    """
    from ollama import chat
    from trading.models import TradingAction

    obs = env.reset()
    initial_equity = obs.account_state.equity

    prompts: List[str] = []
    completions: List[str] = []
    step_rewards: List[float] = []
    equities: List[float] = [initial_equity]
    actions_taken: List[dict] = []
    parse_successes: List[bool] = []

    for step_idx in range(max_steps):
        if obs.done:
            break

        # ── Build observation prompt ──
        obs_text = json.dumps({
            "step": env.state.step_count,
            "market_prices": obs.market_prices,
            "account_state": obs.account_state.model_dump(),
            "last_action_result": obs.result,
            "last_action_error": obs.error,
            "step_reward": obs.reward,
        }, indent=2)

        user_msg = f"Current State:\n{obs_text}\n\nWhat is your next action?"

        # ── Ollama inference ──
        try:
            response = chat(
                model=ollama_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_msg},
                ],
                options={"temperature": temperature},
            )
            llm_reply = response.message.content
        except Exception as e:
            # If Ollama fails, generate a hold action
            llm_reply = '<action>\n{"tool_name": "hold", "tool_args": {}}\n</action>'

        prompts.append(user_msg)
        completions.append(llm_reply)

        # ── Parse action ──
        action_dict = extract_action(llm_reply)
        parsed_ok = action_dict is not None
        parse_successes.append(parsed_ok)

        if not action_dict:
            action_dict = {"tool_name": "hold", "tool_args": {}}

        actions_taken.append(action_dict)

        action = TradingAction(
            tool_name=action_dict.get("tool_name", "hold"),
            tool_args=action_dict.get("tool_args", {}),
        )

        # ── Step the environment ──
        obs = env.step(action)
        step_rewards.append(obs.reward)
        equities.append(obs.account_state.equity)

    # ── Compute episode-level metrics ──
    final_equity = equities[-1]
    pnl = final_equity - initial_equity
    pnl_pct = pnl / initial_equity if initial_equity > 0 else 0.0

    # Max drawdown
    peak = equities[0]
    max_drawdown = 0.0
    for eq in equities:
        if eq > peak:
            peak = eq
        dd = (peak - eq) / peak if peak > 0 else 0.0
        max_drawdown = max(max_drawdown, dd)

    # Risk-adjusted score: higher is better, penalizes drawdowns
    risk_adjusted = max(0.0, 1.0 - max_drawdown)

    # Format compliance score
    format_score = sum(parse_successes) / len(parse_successes) if parse_successes else 0.0

    # Action diversity: unique actions / total steps
    action_labels = [format_action_label(a) for a in actions_taken]
    diversity = len(set(action_labels)) / max(len(action_labels), 1)

    return {
        "prompts": prompts,
        "completions": completions,
        "step_rewards": step_rewards,
        "equities": equities,
        "actions_taken": actions_taken,
        # Episode-level reward signals
        "pnl": pnl,
        "pnl_pct": pnl_pct,
        "risk_adjusted": risk_adjusted,
        "format_score": format_score,
        "diversity": diversity,
        "max_drawdown": max_drawdown,
        "final_equity": final_equity,
        "initial_equity": initial_equity,
        "total_steps": len(step_rewards),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Reward Functions
# ─────────────────────────────────────────────────────────────────────────────

def compute_trading_reward(episode: dict) -> float:
    """Combine multiple reward signals into a single scalar for GRPO.

    Weights:
      - P/L (normalized profit/loss):     2.0
      - Risk-adjusted (drawdown penalty): 1.0
      - Format compliance:                0.5
      - Action diversity:                 0.25
    """
    pnl = episode["pnl_pct"]            # Can be negative
    risk = episode["risk_adjusted"]     # [0, 1]
    fmt = episode["format_score"]       # [0, 1]
    div = episode["diversity"]          # [0, 1]

    return 2.0 * pnl + 1.0 * risk + 0.5 * fmt + 0.25 * div


# ─────────────────────────────────────────────────────────────────────────────
# MLX Log-Probability Computation
# ─────────────────────────────────────────────────────────────────────────────

def calculate_log_probs(model, tokenizer, prompt: str, completion: str):
    """Compute log p(completion | prompt) for GRPO ratio calculation.

    Feeds prompt + completion through the model and sums the log-probabilities
    over the completion tokens only.
    """
    import mlx.core as mx
    import mlx.nn as nn

    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
    completion_tokens = tokenizer.encode(completion, add_special_tokens=False)

    # Truncate if too long for memory (keep last N tokens)
    max_ctx = 2048
    full_tokens = prompt_tokens + completion_tokens
    if len(full_tokens) > max_ctx:
        # Keep the tail, which includes the full completion
        excess = len(full_tokens) - max_ctx
        prompt_tokens = prompt_tokens[excess:]
        full_tokens = prompt_tokens + completion_tokens

    input_ids = mx.array(full_tokens, dtype=mx.int32)[None, :]  # [1, seq_len]
    logits = model(input_ids)  # [1, seq_len, vocab_size]
    log_probs_full = nn.log_softmax(logits, axis=-1)

    prompt_len = len(prompt_tokens)
    completion_len = len(completion_tokens)

    completion_log_probs = []
    for i in range(completion_len):
        pos = prompt_len - 1 + i
        if pos < len(full_tokens) - 1:
            next_token_id = full_tokens[pos + 1]
            log_prob = log_probs_full[0, pos, next_token_id]
            completion_log_probs.append(log_prob)

    if len(completion_log_probs) > 0:
        return mx.sum(mx.stack(completion_log_probs))
    else:
        return mx.array(0.0)


# ─────────────────────────────────────────────────────────────────────────────
# GRPO Trainer
# ─────────────────────────────────────────────────────────────────────────────

class TradingGRPOTrainer:
    """Pure-MLX GRPO trainer for trading on Apple Silicon.

    Architecture:
      - πθ (self.model): MLX model, trainable — receives gradient updates
      - π_ref (self.ref_model): MLX model, frozen — KL penalty anchor
      - Rollouts are generated via Ollama (external, not managed here)
    """

    def __init__(self, model, tokenizer, config: TradingGRPOConfig):
        import mlx.core as mx
        import mlx.nn as nn
        from mlx.utils import tree_flatten, tree_map
        from mlx.optimizers import Adam, cosine_decay, clip_grad_norm

        self.model = model                              # πθ — trainable policy
        self.ref_model = copy.deepcopy(model)            # π_ref — frozen reference
        self.tokenizer = tokenizer
        self.config = config

        # Quantize reference model to save memory (no gradients needed)
        try:
            nn.quantize(self.ref_model, group_size=64, bits=4)
            print("  ✅ Quantized ref_model to 4-bit")
        except Exception as e:
            print(f"  ⚠️  Quantization skipped: {e}")

        # Optimizer with cosine LR schedule + warmup
        total_updates = max(1, config.max_train_steps // config.gradient_accumulation_steps)
        base_lr = cosine_decay(config.learning_rate, total_updates)
        warmup_steps = max(1, int(total_updates * config.warmup_ratio))

        def lr_schedule(step):
            warm = step / warmup_steps if step < warmup_steps else 1.0
            return base_lr(step) * warm

        self.lr_schedule = lr_schedule
        self.optimizer = Adam(learning_rate=lr_schedule)

        # State tracking
        self.step = 0
        self.update_step = 0
        self._accum_grads = None

        # Best model tracking
        self.best_reward = float("-inf")
        self.best_step = 0

        # Logging
        os.makedirs(config.output_dir, exist_ok=True)
        self.log_path = os.path.join(config.output_dir, "training_log.jsonl")
        self.metrics_history: List[dict] = []

    def _log(self, record: dict):
        """Append a metrics record to the log file."""
        self.metrics_history.append(record)
        try:
            with open(self.log_path, "a") as f:
                f.write(json.dumps(record) + "\n")
        except Exception:
            pass

    def _select_training_turns(self, episode: dict) -> List[Tuple[str, str]]:
        """Select representative (prompt, completion) pairs from an episode.

        Samples `turns_to_sample` turns, biased toward more informative steps.
        """
        prompts = episode["prompts"]
        completions = episode["completions"]
        n = len(prompts)

        if n == 0:
            return []

        k = min(self.config.turns_to_sample, n)

        if n <= k:
            return list(zip(prompts, completions))

        # Always include the last turn (carries terminal reward signal)
        # and sample the rest uniformly
        indices = set()
        indices.add(n - 1)       # last step
        indices.add(0)           # first step (initial decision)
        if n > 2:
            indices.add(n // 2)  # mid-episode

        # Fill remaining with random samples
        remaining = list(set(range(n)) - indices)
        random.shuffle(remaining)
        while len(indices) < k and remaining:
            indices.add(remaining.pop())

        sorted_indices = sorted(indices)
        return [(prompts[i], completions[i]) for i in sorted_indices]

    def compute_grpo_loss(self, episodes: List[dict]):
        """Compute GRPO loss over a group of episodes.

        GRPO loss = -E[min(r*A, clip(r, 1-ε, 1+ε)*A)] + β * KL(πθ || π_ref)
        where r = πθ(o|q) / π_ref(o|q)  (using π_ref as proxy for π_old)
        """
        import mlx.core as mx

        # 1. Compute rewards and normalize advantages
        rewards = mx.array([compute_trading_reward(ep) for ep in episodes])
        mean_reward = mx.mean(rewards)
        std_reward = mx.maximum(mx.std(rewards), mx.array(1e-8))
        advantages = (rewards - mean_reward) / std_reward

        # 2. For each episode, use sampled turns for the update
        total_loss = mx.array(0.0)
        total_policy_reward = mx.array(0.0)
        total_kl = mx.array(0.0)
        count = 0

        for idx, ep in enumerate(episodes):
            turns = self._select_training_turns(ep)
            if not turns:
                continue

            adv = advantages[idx]

            for prompt, completion in turns:
                # Log probs under current policy and reference
                current_lp = calculate_log_probs(
                    self.model, self.tokenizer, prompt, completion
                )
                ref_lp = calculate_log_probs(
                    self.ref_model, self.tokenizer, prompt, completion
                )

                # Importance ratio: π_θ / π_ref
                # Since Ollama generates rollouts (not π_θ), we use π_ref as proxy for π_old
                ratio = mx.exp(current_lp - ref_lp)
                clipped_ratio = mx.clip(
                    ratio,
                    1.0 - self.config.clip_eps,
                    1.0 + self.config.clip_eps,
                )
                policy_reward = mx.minimum(ratio * adv, clipped_ratio * adv)

                # KL divergence: D_KL(πθ || π_ref)
                log_ratio_kl = ref_lp - current_lp
                ratio_kl = mx.exp(log_ratio_kl)
                kl_div = ratio_kl - log_ratio_kl - 1

                # Combined objective (negated for minimization)
                objective = policy_reward - self.config.kl_coeff * kl_div
                total_loss = total_loss - objective
                total_policy_reward = total_policy_reward + policy_reward
                total_kl = total_kl + kl_div
                count += 1

        if count > 0:
            total_loss = total_loss / count
            total_policy_reward = total_policy_reward / count
            total_kl = total_kl / count

        return total_loss, float(total_policy_reward), float(total_kl), float(mean_reward)

    def train_step(self, episodes: List[dict]) -> dict:
        """One GRPO training step with gradient accumulation."""
        import mlx.core as mx
        import mlx.nn as nn
        from mlx.utils import tree_map
        from mlx.optimizers import clip_grad_norm

        # Define loss function for MLX's value_and_grad
        def loss_fn():
            loss, _, _, _ = self.compute_grpo_loss(episodes)
            return loss

        # Compute gradients
        grad_fn = nn.value_and_grad(self.model, loss_fn)
        loss, grads = grad_fn()

        # Recompute metrics for logging (without gradient tracking)
        _, policy_reward, kl_div, mean_reward = self.compute_grpo_loss(episodes)

        # Accumulate gradients
        if self._accum_grads is None:
            self._accum_grads = grads
        else:
            self._accum_grads = tree_map(
                lambda a, b: a + b, self._accum_grads, grads
            )

        self.step += 1

        # Apply update when accumulation is complete
        do_update = (self.step % self.config.gradient_accumulation_steps) == 0
        grad_norm_val = None

        if do_update:
            scaled = tree_map(
                lambda g: g / self.config.gradient_accumulation_steps,
                self._accum_grads,
            )
            scaled, grad_norm = clip_grad_norm(scaled, self.config.max_grad_norm)
            self.optimizer.update(self.model, scaled)
            self._accum_grads = None
            self.update_step += 1
            grad_norm_val = float(grad_norm)

            # Force evaluation of updated parameters
            mx.eval(self.model.trainable_parameters())

        return {
            "loss": float(loss),
            "policy_reward": policy_reward,
            "kl_div": kl_div,
            "mean_reward": mean_reward,
            "grad_norm": grad_norm_val,
            "did_update": do_update,
        }

    def save_checkpoint(self, path: str, tag: str = ""):
        """Save model weights and training state."""
        import mlx.nn as nn

        os.makedirs(path, exist_ok=True)

        if isinstance(self.model, nn.Module):
            self.model.save_weights(os.path.join(path, "model.safetensors"))

        # Save training state
        state = {
            "step": self.step,
            "update_step": self.update_step,
            "config": asdict(self.config),
            "best_reward": self.best_reward,
            "best_step": self.best_step,
            "tag": tag,
        }
        with open(os.path.join(path, "trainer_state.json"), "w") as f:
            json.dump(state, f, indent=2)

        print(f"  💾 Checkpoint saved to {path} [{tag}]")

    def train(self, env):
        """Full GRPO training loop with Rich dashboard."""
        import mlx.core as mx
        from rich.console import Console
        from rich.live import Live
        from rich.panel import Panel
        from rich.progress import (
            Progress, SpinnerColumn, TextColumn, BarColumn,
            TimeElapsedColumn, MofNCompleteColumn,
        )
        from rich.table import Table
        from rich.text import Text
        from rich.layout import Layout

        console = Console()

        console.print()
        console.rule("[bold cyan]🤖 GRPO Trading Training Pipeline[/bold cyan]")
        console.print(f"  [dim]Rollout model (Ollama):[/dim]  {self.config.ollama_model}")
        console.print(f"  [dim]Training model (MLX):[/dim]    {self.config.mlx_model}")
        console.print(f"  [dim]Episodes per step:[/dim]       {self.config.num_episodes_per_step}")
        console.print(f"  [dim]Total training steps:[/dim]    {self.config.max_train_steps}")
        console.print(f"  [dim]Steps per episode:[/dim]       {self.config.max_env_steps}")
        console.print(f"  [dim]Asset universe:[/dim]          {self.config.asset_universe}")
        console.print(f"  [dim]Learning rate:[/dim]           {self.config.learning_rate}")
        console.print(f"  [dim]KL coefficient:[/dim]          {self.config.kl_coeff}")
        console.rule()
        console.print()

        mx.random.seed(self.config.seed)
        random.seed(self.config.seed)
        start_time = time.time()

        # Save initial config
        with open(os.path.join(self.config.output_dir, "config.json"), "w") as f:
            json.dump(asdict(self.config), f, indent=2)

        # ── Progress bar ──
        progress = Progress(
            SpinnerColumn(style="cyan"),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=40, complete_style="cyan", finished_style="green"),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
        )
        task_id = progress.add_task(
            "[cyan]Training...", total=self.config.max_train_steps
        )

        all_episode_rewards: List[float] = []  # for final summary

        with progress:
            for step in range(self.config.max_train_steps):
                step_start = time.time()

                # ── 1. Generate episodes via Ollama ──
                progress.update(
                    task_id,
                    description=f"[cyan]Step {step+1}/{self.config.max_train_steps} — rolling out episodes..."
                )

                episodes = []
                for ep_idx in range(self.config.num_episodes_per_step):
                    progress.update(
                        task_id,
                        description=(
                            f"[cyan]Step {step+1}/{self.config.max_train_steps} — "
                            f"episode {ep_idx+1}/{self.config.num_episodes_per_step}"
                        ),
                    )
                    ep = play_trading_episode(
                        env=env,
                        ollama_model=self.config.ollama_model,
                        system_prompt=TRADING_SYSTEM_PROMPT,
                        max_steps=self.config.max_env_steps,
                        temperature=self.config.temperature,
                    )
                    episodes.append(ep)

                # ── 2. GRPO training step ──
                progress.update(
                    task_id,
                    description=f"[yellow]Step {step+1}/{self.config.max_train_steps} — computing GRPO loss..."
                )

                metrics = self.train_step(episodes)
                step_time = time.time() - step_start

                # ── 3. Logging ──
                ep_rewards = [compute_trading_reward(ep) for ep in episodes]
                all_episode_rewards.extend(ep_rewards)

                avg_pnl = sum(ep["pnl"] for ep in episodes) / len(episodes)
                avg_equity = sum(ep["final_equity"] for ep in episodes) / len(episodes)
                avg_drawdown = sum(ep["max_drawdown"] for ep in episodes) / len(episodes)
                avg_format = sum(ep["format_score"] for ep in episodes) / len(episodes)
                avg_steps = sum(ep["total_steps"] for ep in episodes) / len(episodes)

                if metrics["mean_reward"] > self.best_reward:
                    self.best_reward = metrics["mean_reward"]
                    self.best_step = step + 1
                    # Save best model
                    best_path = os.path.join(self.config.output_dir, "best")
                    self.save_checkpoint(best_path, tag=f"best@step-{step+1}")

                mem = psutil.virtual_memory()
                record = {
                    "step": step + 1,
                    "update": self.update_step,
                    "loss": metrics["loss"],
                    "policy_reward": metrics["policy_reward"],
                    "kl": metrics["kl_div"],
                    "mean_reward": metrics["mean_reward"],
                    "avg_pnl": round(avg_pnl, 2),
                    "avg_equity": round(avg_equity, 2),
                    "avg_drawdown_pct": round(avg_drawdown * 100, 1),
                    "avg_format_score": round(avg_format, 3),
                    "avg_steps": round(avg_steps, 1),
                    "step_time_s": round(step_time, 1),
                    "memory_used_gb": round(
                        (mem.total - mem.available) / (1024**3), 1
                    ),
                }
                if metrics["grad_norm"] is not None:
                    record["grad_norm"] = round(metrics["grad_norm"], 1)
                self._log(record)

                # ── Console output ──
                pnl_color = "green" if avg_pnl >= 0 else "red"
                console.print(
                    f"  Step [bold cyan]{step+1:>3}/{self.config.max_train_steps}[/bold cyan] │ "
                    f"Loss: {metrics['loss']:>8.4f} │ "
                    f"Reward: {metrics['mean_reward']:>6.3f} │ "
                    f"KL: {metrics['kl_div']:>6.4f} │ "
                    f"P/L: [{pnl_color}]${avg_pnl:>+8.2f}[/{pnl_color}] │ "
                    f"Equity: ${avg_equity:>10,.2f} │ "
                    f"DD: {avg_drawdown*100:>5.1f}% │ "
                    f"Fmt: {avg_format:>4.0%} │ "
                    f"Time: {step_time:>6.1f}s │ "
                    f"Mem: {record['memory_used_gb']:.1f}GB"
                )

                # ── 4. Periodic checkpoint ──
                if (step + 1) % self.config.save_steps == 0:
                    ckpt_path = os.path.join(
                        self.config.output_dir, f"checkpoint-{step+1}"
                    )
                    self.save_checkpoint(ckpt_path, tag=f"step-{step+1}")

                progress.update(task_id, advance=1)

        # ── 5. Save final model ──
        total_time = time.time() - start_time
        final_path = os.path.join(self.config.output_dir, "final")
        self.save_checkpoint(final_path, tag="final")

        # ── 6. Print training summary ──
        console.print()
        console.rule("[bold magenta]🏁 Training Complete[/bold magenta]")

        summary_table = Table(
            title="📋 GRPO Training Summary",
            show_header=True,
            header_style="bold white on dark_blue",
            border_style="bright_blue",
            title_style="bold bright_cyan",
            expand=True,
            padding=(0, 2),
        )
        summary_table.add_column("Metric", style="bold", justify="left", min_width=25)
        summary_table.add_column("Value", justify="right", min_width=20)

        summary_table.add_row("Total Training Time", f"{total_time/60:.1f} minutes")
        summary_table.add_row("Total Steps", str(self.config.max_train_steps))
        summary_table.add_row("Optimizer Updates", str(self.update_step))
        summary_table.add_row("Episodes Generated", str(len(all_episode_rewards)))
        summary_table.add_row("", "")
        summary_table.add_row(
            "Best Reward",
            f"[green]{self.best_reward:.4f}[/green] (step {self.best_step})"
        )
        if self.metrics_history:
            final = self.metrics_history[-1]
            summary_table.add_row("Final Loss", f"{final['loss']:.4f}")
            summary_table.add_row("Final KL", f"{final['kl']:.4f}")
            summary_table.add_row("Final Avg P/L", f"${final['avg_pnl']:+.2f}")
            summary_table.add_row("Final Avg Equity", f"${final['avg_equity']:,.2f}")
        summary_table.add_row("", "")
        summary_table.add_row("Best Model", os.path.join(self.config.output_dir, "best"))
        summary_table.add_row("Final Model", final_path)
        summary_table.add_row("Training Log", self.log_path)

        console.print(summary_table)
        console.print()

        return self.metrics_history


# ─────────────────────────────────────────────────────────────────────────────
# Post-Training Visualization
# ─────────────────────────────────────────────────────────────────────────────

def generate_training_report(metrics_history: List[dict], output_dir: str):
    """Generate an interactive Plotly training dashboard."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    if not metrics_history:
        return

    steps = [m["step"] for m in metrics_history]
    rewards = [m["mean_reward"] for m in metrics_history]
    losses = [m["loss"] for m in metrics_history]
    pnls = [m["avg_pnl"] for m in metrics_history]
    equities = [m["avg_equity"] for m in metrics_history]
    drawdowns = [m.get("avg_drawdown_pct", 0) for m in metrics_history]
    kls = [m["kl"] for m in metrics_history]

    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            "Mean Reward", "GRPO Loss",
            "Avg P/L ($)", "Avg Final Equity ($)",
            "KL Divergence", "Max Drawdown (%)",
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.1,
    )

    # Row 1
    fig.add_trace(
        go.Scatter(x=steps, y=rewards, name="Reward",
                   line=dict(color="#00d4ff", width=2)),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=steps, y=losses, name="Loss",
                   line=dict(color="#ff6b6b", width=2)),
        row=1, col=2,
    )

    # Row 2
    pnl_colors = ["rgba(0,230,118,0.7)" if p >= 0 else "rgba(255,82,82,0.7)" for p in pnls]
    fig.add_trace(
        go.Bar(x=steps, y=pnls, name="P/L", marker_color=pnl_colors),
        row=2, col=1,
    )
    fig.add_trace(
        go.Scatter(x=steps, y=equities, name="Equity",
                   line=dict(color="#ffa726", width=2)),
        row=2, col=2,
    )

    # Row 3
    fig.add_trace(
        go.Scatter(x=steps, y=kls, name="KL",
                   line=dict(color="#ab47bc", width=2)),
        row=3, col=1,
    )
    fig.add_trace(
        go.Scatter(x=steps, y=drawdowns, name="Drawdown",
                   line=dict(color="#ef5350", width=2), fill="tozeroy",
                   fillcolor="rgba(239,83,80,0.2)"),
        row=3, col=2,
    )

    fig.update_layout(
        title=dict(
            text="🤖 GRPO Trading Training — Dashboard",
            font=dict(size=20, color="#ffffff"),
        ),
        template="plotly_dark",
        paper_bgcolor="#0d1117",
        plot_bgcolor="#161b22",
        font=dict(family="Inter, sans-serif", color="#c9d1d9"),
        showlegend=False,
        height=900,
        margin=dict(l=60, r=40, t=80, b=40),
    )

    output_file = os.path.join(output_dir, "training_report.html")
    fig.write_html(output_file, include_plotlyjs="cdn")
    print(f"\n📊 Training report saved: {os.path.abspath(output_file)}")
    print("   Open in your browser to explore training dynamics!\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="GRPO Trading Training — Local Ollama + MLX"
    )
    parser.add_argument(
        "--max-train-steps", type=int, default=None,
        help="Total GRPO training steps (default: 30)",
    )
    parser.add_argument(
        "--num-episodes", type=int, default=None,
        help="Episodes per training step / group size G (default: 4)",
    )
    parser.add_argument(
        "--max-env-steps", type=int, default=None,
        help="Max steps per trading episode (default: 100)",
    )
    parser.add_argument(
        "--ollama-model", type=str, default=None,
        help="Ollama model for rollouts (default: lfm2.5-thinking)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory for checkpoints (default: trading-grpo-lfm2.5)",
    )
    parser.add_argument(
        "--lr", type=float, default=None,
        help="Learning rate (default: 3e-6)",
    )
    args = parser.parse_args()

    # ── Build config ──
    config = TradingGRPOConfig()
    if args.max_train_steps is not None:
        config.max_train_steps = args.max_train_steps
    if args.num_episodes is not None:
        config.num_episodes_per_step = args.num_episodes
    if args.max_env_steps is not None:
        config.max_env_steps = args.max_env_steps
    if args.ollama_model is not None:
        config.ollama_model = args.ollama_model
    if args.output_dir is not None:
        config.output_dir = args.output_dir
    if args.lr is not None:
        config.learning_rate = args.lr

    # ── Verify prerequisites ──
    from rich.console import Console
    console = Console()

    console.print()
    console.rule("[bold cyan]🔧 Pre-flight Checks[/bold cyan]")

    # 1. Verify MLX / Apple Silicon
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten

    assert mx.metal.is_available(), "Metal GPU not available — MLX requires Apple Silicon!"
    mem = psutil.virtual_memory()
    console.print(
        f"  ✅ MLX backend: Metal ({mx.default_device()}) — "
        f"{mem.total / (1024**3):.1f} GB unified memory"
    )

    # 2. Verify Ollama
    try:
        from ollama import chat
        test_resp = chat(
            model=config.ollama_model,
            messages=[{"role": "user", "content": "Say OK"}],
            options={"num_predict": 5},
        )
        console.print(f"  ✅ Ollama model: {config.ollama_model}")
    except Exception as e:
        console.print(f"  [bold red]❌ Ollama check failed: {e}[/bold red]")
        console.print(
            "  [dim]Make sure Ollama is running and the model is pulled: "
            f"ollama pull {config.ollama_model}[/dim]"
        )
        sys.exit(1)

    # 3. Load MLX model
    console.print(f"  ⏳ Loading MLX model: {config.mlx_model}...")
    from mlx_lm import load as mlx_load

    model, tokenizer = mlx_load(
        config.mlx_model,
        tokenizer_config={"trust_remote_code": True},
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    num_params = sum(p.size for _, p in tree_flatten(model.parameters()))
    console.print(
        f"  ✅ MLX model loaded: {num_params / 1e6:.1f}M parameters"
    )

    # 4. Initialize trading environment
    console.print("  ⏳ Initializing TradingEnvironment (backtest mode)...")
    os.environ["ENV_MODE"] = "backtest"
    os.environ["ASSET_UNIVERSE"] = config.asset_universe
    os.environ["INITIAL_VIRTUAL_CASH"] = str(config.initial_cash)
    os.environ["MAX_STEPS"] = str(config.max_env_steps)

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from trading.server.trading_environment import TradingEnvironment

    env = TradingEnvironment()
    console.print(
        f"  ✅ Environment ready: {config.asset_universe} "
        f"(backtest, {config.max_env_steps} steps/episode)"
    )

    console.rule()
    console.print()

    # ── Launch training ──
    trainer = TradingGRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        config=config,
    )

    metrics = trainer.train(env)

    # ── Generate visualization ──
    generate_training_report(metrics, config.output_dir)


if __name__ == "__main__":
    main()
