"""Minimal VecTask abstraction for task skeleton development."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class StepResult:
    obs: list[list[float]]
    rewards: list[float]
    dones: list[bool]
    infos: dict[str, Any]


class VecTask:
    """A lightweight stand-in for IsaacGymEnvs VecTask."""

    def __init__(self, cfg: dict[str, Any]):
        self.cfg = cfg
        self.num_envs = int(cfg.get("num_envs", 1))
        self.num_obs = int(cfg.get("num_observations", 8))
        self.num_actions = int(cfg.get("num_actions", 12))

    def reset(self) -> list[list[float]]:
        raise NotImplementedError

    def step(self, actions: list[list[float]]) -> StepResult:
        raise NotImplementedError
