"""Go2Bridge task with GO2 actor creation and minimal simulation wiring."""

from __future__ import annotations

import os
from typing import Any

import torch
from isaacgym import gymapi, gymtorch

from isaacgymenvs.tasks.base.vec_task import StepResult, VecTask


class Go2Bridge(VecTask):
    def __init__(self, cfg: dict[str, Any]):
        super().__init__(cfg)

        self.bridge_type = cfg.get("bridge_type", "narrow_static_bridge")
        self.max_episode_length = int(cfg.get("max_episode_length", 500))
        self.dt = float(cfg.get("dt", 0.02))
        self.control_dt = float(cfg.get("control_dt", self.dt))

        self.gym = gymapi.acquire_gym()
        self.sim = None
        self.envs: list[Any] = []
        self.actor_handles: list[int] = []

        self.progress_buf = [0 for _ in range(self.num_envs)]
        self.obs_buf = [[0.0 for _ in range(self.num_obs)] for _ in range(self.num_envs)]
        self.rew_buf = [0.0 for _ in range(self.num_envs)]
        self.reset_buf = [False for _ in range(self.num_envs)]

        self.default_dof_pos = None
        self.actor_indices = None
        self.root_states = None
        self.dof_state = None
        self.dof_pos = None
        self.dof_vel = None
        self.num_dof = 0
        self.num_bodies = 0
        self.num_joints = 0

        self._build_static_terrain()
        self.create_sim()
        self._acquire_tensors()
        self.reset_idx(torch.arange(self.num_envs, dtype=torch.int32))

    def _build_static_terrain(self) -> None:
        if self.bridge_type == "fixed_plane":
            self.terrain_desc = {"kind": "plane", "width": None}
        else:
            self.terrain_desc = {
                "kind": "bridge",
                "width": float(self.cfg.get("bridge_width", 0.35)),
                "length": float(self.cfg.get("bridge_length", 3.0)),
            }

    def _resolve_asset_path(self) -> tuple[str, str]:
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        candidates = [
            os.path.join(repo_root, "assets", "go2", "go2.urdf"),
            os.path.join(repo_root, "isaacgymenvs", "assets", "go2", "go2.urdf"),
        ]
        for urdf_path in candidates:
            if os.path.isfile(urdf_path):
                return os.path.dirname(urdf_path), os.path.basename(urdf_path)

        requested = os.path.join("assets", "go2", "go2.urdf")
        raise FileNotFoundError(
            f"Unable to find GO2 asset '{requested}'. Tried: {candidates}"
        )

    def create_sim(self) -> None:
        sim_params = gymapi.SimParams()
        sim_params.dt = self.control_dt
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
        sim_params.substeps = int(self.cfg.get("substeps", 2))

        physics_engine = gymapi.SIM_PHYSX
        sim_device = self.cfg.get("sim_device", "cpu")
        graphics_device_id = int(self.cfg.get("graphics_device_id", 0))

        self.sim = self.gym.create_sim(0, graphics_device_id, physics_engine, sim_params)
        if self.sim is None:
            raise RuntimeError("Failed to create Isaac Gym simulation")

        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

        self._create_envs()

    def _create_envs(self) -> None:
        asset_root, asset_file = self._resolve_asset_path()
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = False
        asset_options.disable_gravity = False
        asset_options.flip_visual_attachments = False
        asset_options.use_mesh_materials = True

        go2_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        if go2_asset is None:
            raise RuntimeError(f"Failed to load asset: {os.path.join(asset_root, asset_file)}")

        self.num_dof = self.gym.get_asset_dof_count(go2_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(go2_asset)
        self.num_joints = self.gym.get_asset_joint_count(go2_asset)

        print(f"[Go2Bridge] Asset loaded: {os.path.join(asset_root, asset_file)}")
        print(f"[Go2Bridge] num_dof={self.num_dof}")
        print(f"[Go2Bridge] num_bodies={self.num_bodies}")
        print(f"[Go2Bridge] num_joints={self.num_joints}")

        dof_props = self.gym.get_asset_dof_properties(go2_asset)
        dof_props["driveMode"][:] = gymapi.DOF_MODE_POS
        dof_props["stiffness"][:] = 60.0
        dof_props["damping"][:] = 2.0
        dof_props["effort"][:] = 33.5
        dof_props["velocity"][:] = 30.0

        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float32)
        stand_pose = torch.tensor([0.0, 0.8, -1.5] * 4, dtype=torch.float32)
        self.default_dof_pos[:] = stand_pose[: self.num_dof]

        env_spacing = float(self.cfg.get("env_spacing", 2.0))
        env_lower = gymapi.Vec3(-env_spacing, -env_spacing, 0.0)
        env_upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)
        num_per_row = int(self.cfg.get("num_per_row", int(self.num_envs**0.5) + 1))

        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(0.0, 0.0, 0.4)
        start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        for i in range(self.num_envs):
            env_ptr = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)
            actor_handle = self.gym.create_actor(env_ptr, go2_asset, start_pose, "go2", i, 1, 0)
            self.gym.set_actor_dof_properties(env_ptr, actor_handle, dof_props)
            self.gym.set_actor_dof_position_targets(
                env_ptr, actor_handle, self.default_dof_pos.cpu().numpy()
            )

            self.envs.append(env_ptr)
            self.actor_handles.append(actor_handle)

        self.actor_indices = torch.arange(self.num_envs, dtype=torch.int32)

    def _acquire_tensors(self) -> None:
        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)

        self.root_states = gymtorch.wrap_tensor(actor_root_state_tensor)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]

    def reset_idx(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return

        ids_long = env_ids.to(dtype=torch.long)
        self.root_states[ids_long, 0:3] = torch.tensor([0.0, 0.0, 0.4], dtype=torch.float32)
        self.root_states[ids_long, 3:7] = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float32)
        self.root_states[ids_long, 7:13] = 0.0

        self.dof_pos[ids_long] = self.default_dof_pos
        self.dof_vel[ids_long] = 0.0

        actor_ids = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(actor_ids),
            len(actor_ids),
        )

        flat_dof_state = self.dof_state.view(-1, 2)
        dof_indices = (
            ids_long.unsqueeze(1) * self.num_dof
            + torch.arange(self.num_dof, dtype=torch.long).unsqueeze(0)
        ).reshape(-1)
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(flat_dof_state),
            gymtorch.unwrap_tensor(dof_indices.to(dtype=torch.int32)),
            len(dof_indices),
        )

        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.dof_pos))

        for env_id in actor_ids.tolist():
            self.progress_buf[env_id] = 0
            self.rew_buf[env_id] = 0.0
            self.reset_buf[env_id] = False

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
        self.reset_idx(torch.arange(self.num_envs, dtype=torch.int32))
        return self._compute_observation()

    def step(self, actions: list[list[float]]) -> StepResult:
        if len(actions) != self.num_envs:
            raise ValueError(f"actions env dimension must be {self.num_envs}, got {len(actions)}")
        for row in actions:
            if len(row) != self.num_actions:
                raise ValueError(
                    f"actions action dimension must be {self.num_actions}, got {len(row)}"
                )

        action_tensor = torch.tensor(actions, dtype=torch.float32)
        if self.num_dof > 0:
            target = self.default_dof_pos.unsqueeze(0) + 0.25 * action_tensor[:, : self.num_dof]
            self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(target.contiguous()))

        for env_id in range(self.num_envs):
            self.progress_buf[env_id] += 1
            control_penalty = sum(a * a for a in actions[env_id]) / self.num_actions
            self.rew_buf[env_id] = 1.0 - 0.05 * control_penalty
            self.reset_buf[env_id] = self.progress_buf[env_id] >= self.max_episode_length

        done_envs = [i for i, done in enumerate(self.reset_buf) if done]
        if done_envs:
            self.reset_idx(torch.tensor(done_envs, dtype=torch.int32))

        obs = self._compute_observation()
        infos: dict[str, Any] = {
            "terrain": self.terrain_desc,
            "progress": self.progress_buf[:],
        }
        return StepResult(obs=obs, rewards=self.rew_buf[:], dones=self.reset_buf[:], infos=infos)
