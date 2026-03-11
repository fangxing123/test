"""Go2Bridge task skeleton with static terrain placeholder."""

from __future__ import annotations

from typing import Any

from isaacgymenvs.tasks.base.vec_task import StepResult, VecTask


class Go2Bridge(VecTask):
    def __init__(self, cfg: dict[str, Any]):
        super().__init__(cfg)
        self.bridge_type = cfg.get("bridge_type", "narrow_static_bridge")
        self.max_episode_length = int(cfg.get("max_episode_length", 500))
        self.dt = float(cfg.get("dt", 0.02))

        self.progress_buf = [0 for _ in range(self.num_envs)]
        self.obs_buf = [[0.0 for _ in range(self.num_obs)] for _ in range(self.num_envs)]
        self.rew_buf = [0.0 for _ in range(self.num_envs)]
        self.reset_buf = [False for _ in range(self.num_envs)]

        self._build_static_terrain()

    def _build_static_terrain(self) -> None:
        if self.bridge_type == "fixed_plane":
            self.terrain_desc = {"kind": "plane", "width": None}
        else:
            self.terrain_desc = {
                "kind": "bridge",
                "width": float(self.cfg.get("bridge_width", 0.35)),
                "length": float(self.cfg.get("bridge_length", 3.0)),
            }

    def _compute_observation(self) -> list[list[float]]:
        for env_id in range(self.num_envs):
            row = self.obs_buf[env_id]
            for i in range(self.num_obs):
                row[i] = 0.0
            row[0] = self.progress_buf[env_id] * self.dt
            if self.num_obs > 1:
                row[1] = 1.0 if self.terrain_desc["kind"] == "bridge" else 0.0
        return [r[:] for r in self.obs_buf]

    def reset(self) -> list[list[float]]:
        for env_id in range(self.num_envs):
            self.progress_buf[env_id] = 0
            self.rew_buf[env_id] = 0.0
            self.reset_buf[env_id] = False
        return self._compute_observation()

    def step(self, actions: list[list[float]]) -> StepResult:
        if len(actions) != self.num_envs:
            raise ValueError(f"actions env dimension must be {self.num_envs}, got {len(actions)}")
        for row in actions:
            if len(row) != self.num_actions:
                raise ValueError(
                    f"actions action dimension must be {self.num_actions}, got {len(row)}"
                )

        for env_id in range(self.num_envs):
            self.progress_buf[env_id] += 1
            control_penalty = sum(a * a for a in actions[env_id]) / self.num_actions
            self.rew_buf[env_id] = 1.0 - 0.05 * control_penalty
            self.reset_buf[env_id] = self.progress_buf[env_id] >= self.max_episode_length

        obs = self._compute_observation()
        infos: dict[str, Any] = {
            "terrain": self.terrain_desc,
            "progress": self.progress_buf[:],
        }
        return StepResult(obs=obs, rewards=self.rew_buf[:], dones=self.reset_buf[:], infos=infos)
