#!/usr/bin/env python3
import bootstrap  # ensure JAX env before imports
"""End-to-end integration check for MC-driven RL pipeline."""

try:
    import torch
except Exception:  # pragma: no cover - optional dependency
    torch = None

from engines.mc.jax_backend import ensure_jax, jax
from train_transformer_gpu import (
    ACTION_SPACE,
    MCRLEnvironment,
    PolicyNet,
    TrainConfig,
    _synthetic_price_series,
    TORCH_AVAILABLE,
)


def verify() -> None:
    ensure_jax()
    devices = []
    try:
        devices = jax.devices() if jax is not None else []
    except Exception:
        devices = []

    cfg = TrainConfig(episodes=1, steps_per_episode=32)
    prices = _synthetic_price_series(length=cfg.steps_per_episode + 64, seed=cfg.seed + 7)
    env = MCRLEnvironment(prices, cfg)
    initial_state = env.reset()

    if TORCH_AVAILABLE:
        policy = PolicyNet(env.state_dim).to(cfg.device)
        policy.eval()

        state = torch.tensor(initial_state, dtype=torch.float32, device=cfg.device)
        with torch.no_grad():
            logits = policy(state)
            probs = torch.softmax(logits, dim=-1)
            action_idx = torch.argmax(probs).item()
        action = ACTION_SPACE[action_idx]
    else:
        # Torch 없는 환경에서는 가격 모멘텀 기반 휴리스틱으로 행동을 선택하고,
        # MC 시뮬레이션은 env.step 내부에서 실행하도록 위임한다.
        price_change = float(initial_state[0])
        action = ACTION_SPACE[2] if price_change >= 0 else ACTION_SPACE[0]

    next_state, reward, done, info = env.step(action)

    print("✅ Integration OK")
    print(f"JAX devices: {devices}")
    print(f"PyTorch available: {TORCH_AVAILABLE}")
    print(f"Action chosen: {action} | EV={info.get('ev'):.6f} | cost={info.get('cost'):.4f} | reward={reward:.6f}")
    print(f"Next state sample: {next_state[:3]}")
    print(f"Done? {done}")


if __name__ == "__main__":
    verify()
