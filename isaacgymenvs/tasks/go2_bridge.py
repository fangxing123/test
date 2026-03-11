import os
from typing import Dict, Tuple

import numpy as np
import torch

from isaacgym import gymapi, gymtorch, torch_utils
from isaacgym.torch_utils import *
from isaacgymenvs.tasks.base.vec_task import VecTask


class Go2Bridge(VecTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.cfg = cfg

        self.dt = self.cfg["sim"]["dt"]
        self.max_episode_length = self.cfg["env"]["episodeLength"]
        self.action_scale = self.cfg["env"]["control"]["actionScale"]

        self.base_init_state = self.cfg["env"]["baseInitState"]
        self.default_dof_pos = self.cfg["env"]["defaultJointAngles"]
        self.command_ranges = self.cfg["env"]["commandRanges"]

        self.reward_scales = self.cfg["env"]["learn"]

        self.up_axis = "z"
        self.up_axis_idx = 2

        self.feet_names = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
        self.bridge_segment_names = [f"plank_{i}" for i in range(10)]

        self.num_actuated_dof = 12
        self.cfg["env"]["numActions"] = self.num_actuated_dof

        # obs: lin vel(3), ang vel(3), gravity(3), dof pos(12), dof vel(12), prev actions(12), cmd(3), foot force(12), bridge summary(12)
        self.cfg["env"]["numObservations"] = 72

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id,
                         headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)

        self.root_states = gymtorch.wrap_tensor(actor_root_state).view(self.num_envs, self.num_actors, 13)
        self.robot_root_states = self.root_states[:, 0]
        self.bridge_root_states = self.root_states[:, 1]

        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor).view(self.num_envs, self.num_dof, 2)
        self.dof_pos = self.dof_state[:, :self.num_actuated_dof, 0]
        self.dof_vel = self.dof_state[:, :self.num_actuated_dof, 1]

        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state).view(self.num_envs, self.num_bodies, 13)
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, self.num_bodies, 3)
        self.force_sensor_tensor = gymtorch.wrap_tensor(sensor_tensor).view(self.num_envs, len(self.feet_indices), 6)

        self.actions = torch.zeros((self.num_envs, self.num_actions), device=self.device, dtype=torch.float)
        self.last_actions = torch.zeros_like(self.actions)
        self.commands = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float)

        self.base_lin_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float)
        self.base_ang_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float)
        self.projected_gravity = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float)

        self.base_init_state_tensor = to_torch([
            self.base_init_state["pos"][0],
            self.base_init_state["pos"][1],
            self.base_init_state["pos"][2],
            self.base_init_state["rot"][0],
            self.base_init_state["rot"][1],
            self.base_init_state["rot"][2],
            self.base_init_state["rot"][3],
            self.base_init_state["linVel"][0],
            self.base_init_state["linVel"][1],
            self.base_init_state["linVel"][2],
            self.base_init_state["angVel"][0],
            self.base_init_state["angVel"][1],
            self.base_init_state["angVel"][2],
        ], device=self.device)

        self.default_dof_pos_tensor = torch.zeros(self.num_actuated_dof, device=self.device, dtype=torch.float)
        for i, name in enumerate(self.dof_names):
            self.default_dof_pos_tensor[i] = self.default_dof_pos[name]

        self.gravity_vec = to_torch(get_axis_params(-1.0, self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))

        self.initial_root_states = self.root_states.clone()
        self.initial_dof_pos = self.default_dof_pos_tensor.unsqueeze(0).repeat(self.num_envs, 1)
        self.initial_dof_vel = torch.zeros_like(self.initial_dof_pos)

        self.extras = {}
        self.reset_idx(torch.arange(self.num_envs, device=self.device))

    def create_sim(self):
        self.up_axis_idx = 2
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)

        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = 1.0
        plane_params.dynamic_friction = 1.0
        self.gym.add_ground(self.sim, plane_params)

        self._create_envs(self.num_envs, self.cfg["env"]["envSpacing"], int(np.sqrt(self.num_envs)))

    def _create_envs(self, num_envs, spacing, num_per_row):
        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../assets")
        robot_asset_file = "go2/go2.urdf"
        bridge_asset_file = "bridge/floating_bridge.urdf"

        robot_asset_options = gymapi.AssetOptions()
        robot_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        robot_asset_options.collapse_fixed_joints = False
        robot_asset_options.replace_cylinder_with_capsule = True
        robot_asset_options.flip_visual_attachments = False
        robot_asset_options.fix_base_link = False
        robot_asset_options.disable_gravity = False
        robot_asset_options.use_mesh_materials = True

        bridge_asset_options = gymapi.AssetOptions()
        bridge_asset_options.fix_base_link = False
        bridge_asset_options.disable_gravity = False
        bridge_asset_options.collapse_fixed_joints = False
        bridge_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE

        robot_asset = self.gym.load_asset(self.sim, asset_root, robot_asset_file, robot_asset_options)
        bridge_asset = self.gym.load_asset(self.sim, asset_root, bridge_asset_file, bridge_asset_options)

        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset) + self.gym.get_asset_rigid_body_count(bridge_asset)

        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        robot_body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        bridge_body_names = self.gym.get_asset_rigid_body_names(bridge_asset)

        self.robot_dof_props = self.gym.get_asset_dof_properties(robot_asset)
        self.bridge_dof_props = self.gym.get_asset_dof_properties(bridge_asset)

        for i in range(self.num_dof):
            self.robot_dof_props["stiffness"][i] = self.cfg["env"]["control"]["stiffness"]
            self.robot_dof_props["damping"][i] = self.cfg["env"]["control"]["damping"]

        for i in range(len(self.bridge_dof_props)):
            self.bridge_dof_props["damping"][i] = self.cfg["env"]["bridge"]["jointDamping"]
            self.bridge_dof_props["friction"][i] = self.cfg["env"]["bridge"]["jointFriction"]
            self.bridge_dof_props["stiffness"][i] = 0.0

        feet_indices_asset = []
        for feet_name in self.feet_names:
            feet_indices_asset.append(self.gym.find_asset_rigid_body_index(robot_asset, feet_name))
            sensor_pose = gymapi.Transform()
            self.gym.create_asset_force_sensor(robot_asset, feet_indices_asset[-1], sensor_pose)

        self.robot_start_pose = gymapi.Transform()
        self.robot_start_pose.p = gymapi.Vec3(self.base_init_state["pos"][0], self.base_init_state["pos"][1], self.base_init_state["pos"][2])
        self.robot_start_pose.r = gymapi.Quat(
            self.base_init_state["rot"][0], self.base_init_state["rot"][1], self.base_init_state["rot"][2], self.base_init_state["rot"][3]
        )

        self.bridge_start_pose = gymapi.Transform()
        self.bridge_start_pose.p = gymapi.Vec3(0.0, 0.0, self.cfg["env"]["bridge"]["height"])

        env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)
        self.envs = []
        self.robot_handles = []
        self.bridge_handles = []

        self.feet_indices = []
        self.bridge_segment_indices = []
        self.robot_base_idx = self.gym.find_asset_rigid_body_index(robot_asset, "base")

        for i in range(num_envs):
            env_ptr = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)

            robot_handle = self.gym.create_actor(env_ptr, robot_asset, self.robot_start_pose, "go2", i, 1, 0)
            bridge_handle = self.gym.create_actor(env_ptr, bridge_asset, self.bridge_start_pose, "bridge", i, 0, 0)

            self.gym.set_actor_dof_properties(env_ptr, robot_handle, self.robot_dof_props)
            self.gym.set_actor_dof_properties(env_ptr, bridge_handle, self.bridge_dof_props)

            robot_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, robot_handle)
            bridge_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, bridge_handle)

            for p in robot_shape_props:
                p.friction = self.cfg["env"]["terrain"]["friction"]
                p.restitution = 0.0

            for p in bridge_shape_props:
                p.friction = self.cfg["env"]["bridge"]["friction"]
                p.restitution = 0.0

            self.gym.set_actor_rigid_shape_properties(env_ptr, robot_handle, robot_shape_props)
            self.gym.set_actor_rigid_shape_properties(env_ptr, bridge_handle, bridge_shape_props)

            self.envs.append(env_ptr)
            self.robot_handles.append(robot_handle)
            self.bridge_handles.append(bridge_handle)

            feet_env_indices = []
            for name in self.feet_names:
                feet_env_indices.append(self.gym.find_actor_rigid_body_handle(env_ptr, robot_handle, name))
            self.feet_indices.append(feet_env_indices)

            seg_indices = []
            for bname in self.bridge_segment_names:
                idx = self.gym.find_actor_rigid_body_handle(env_ptr, bridge_handle, bname)
                if idx == -1:
                    idx = self.gym.find_actor_rigid_body_handle(env_ptr, bridge_handle, bridge_body_names[min(len(seg_indices), len(bridge_body_names)-1)])
                seg_indices.append(idx)
            self.bridge_segment_indices.append(seg_indices)

        self.num_actors = 2
        self.feet_indices = torch.tensor(self.feet_indices, device=self.device, dtype=torch.long)
        self.bridge_segment_indices = torch.tensor(self.bridge_segment_indices, device=self.device, dtype=torch.long)

    def compute_observations(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)

        base_quat = self.robot_root_states[:, 3:7]
        inv_base_quat = quat_conjugate(base_quat)

        self.base_lin_vel = quat_rotate_inverse(base_quat, self.robot_root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(base_quat, self.robot_root_states[:, 10:13])
        self.projected_gravity = quat_rotate(inv_base_quat, self.gravity_vec)

        dof_pos_scaled = self.dof_pos - self.default_dof_pos_tensor.unsqueeze(0)
        dof_vel_scaled = self.dof_vel

        foot_forces = self.force_sensor_tensor[:, :, :3].reshape(self.num_envs, -1)

        bridge_summary = self._compute_bridge_summary(base_quat)

        self.obs_buf = torch.cat((
            self.base_lin_vel,
            self.base_ang_vel,
            self.projected_gravity,
            dof_pos_scaled,
            dof_vel_scaled,
            self.last_actions,
            self.commands,
            foot_forces,
            bridge_summary,
        ), dim=-1)

        return self.obs_buf

    def _compute_bridge_summary(self, base_quat):
        base_pos = self.robot_root_states[:, 0:3]
        seg_pos = self.rigid_body_states[:, self.bridge_segment_indices, 0:3]
        seg_quat = self.rigid_body_states[:, self.bridge_segment_indices, 3:7]

        dists = torch.norm(seg_pos - base_pos.unsqueeze(1), dim=-1)
        _, idx = torch.topk(dists, k=4, dim=1, largest=False)

        gather_idx = idx.unsqueeze(-1).repeat(1, 1, 3)
        near_seg_pos = torch.gather(seg_pos, 1, gather_idx)

        quat_idx = idx.unsqueeze(-1).repeat(1, 1, 4)
        near_seg_quat = torch.gather(seg_quat, 1, quat_idx)

        rel_pos = near_seg_pos - base_pos.unsqueeze(1)
        rel_pos_local = quat_rotate_inverse(base_quat.unsqueeze(1).repeat(1, 4, 1).reshape(-1, 4), rel_pos.reshape(-1, 3)).reshape(self.num_envs, 4, 3)

        seg_euler = get_euler_xyz(near_seg_quat.reshape(-1, 4))
        pitch_roll = torch.stack((seg_euler[1], seg_euler[0]), dim=-1).reshape(self.num_envs, 4, 2)

        summary = torch.cat((rel_pos_local[:, :, 2:3], pitch_roll), dim=-1).reshape(self.num_envs, -1)
        return summary

    def compute_reward(self):
        base_quat = self.robot_root_states[:, 3:7]
        heading_vel = quat_rotate_inverse(base_quat, self.robot_root_states[:, 7:10])[:, 0]

        cmd_xy = self.commands[:, :2]
        vel_xy = quat_rotate_inverse(base_quat, self.robot_root_states[:, 7:10])[:, :2]
        cmd_tracking = torch.exp(-torch.sum((vel_xy - cmd_xy) ** 2, dim=-1) / 0.25)

        forward_progress = heading_vel

        bridge_center_y = self.bridge_root_states[:, 1]
        on_bridge = torch.exp(-torch.square(self.robot_root_states[:, 1] - bridge_center_y) / 0.2)

        upright = torch.square(self.projected_gravity[:, 2]).clamp(0.0, 1.0)

        action_smooth = -torch.sum(torch.square(self.actions - self.last_actions), dim=-1)

        foot_vel = self.rigid_body_states[:, self.feet_indices, 7:10]
        foot_speed = torch.norm(foot_vel, dim=-1)
        contact_mag = torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1)
        slip = torch.sum((contact_mag > 5.0) * foot_speed, dim=-1)

        fall = torch.where(
            (self.robot_root_states[:, 2] < self.cfg["env"]["termination"]["minBaseHeight"]) |
            (upright < self.cfg["env"]["termination"]["minUpright"]),
            torch.ones_like(upright),
            torch.zeros_like(upright)
        )

        self.rew_buf[:] = (
            self.reward_scales["commandTracking"] * cmd_tracking
            + self.reward_scales["forwardProgress"] * forward_progress
            + self.reward_scales["onBridge"] * on_bridge
            + self.reward_scales["upright"] * upright
            + self.reward_scales["actionSmooth"] * action_smooth
            + self.reward_scales["footSlip"] * (-slip)
            + self.reward_scales["fallPenalty"] * (-fall)
        )

        self.reset_buf = torch.where(fall > 0.0, torch.ones_like(self.reset_buf), self.reset_buf)
        self.reset_buf = torch.where(self.progress_buf >= self.max_episode_length - 1, torch.ones_like(self.reset_buf), self.reset_buf)

    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return

        env_ids_int32 = env_ids.to(dtype=torch.int32)

        self.root_states[env_ids, 0] = self.initial_root_states[env_ids, 0]
        noise_xy = (torch.rand((len(env_ids), 2), device=self.device) - 0.5) * 0.2
        self.root_states[env_ids, 0, 0:2] += noise_xy

        self.root_states[env_ids, 1] = self.initial_root_states[env_ids, 1]

        self.dof_pos[env_ids] = self.initial_dof_pos[env_ids] + 0.05 * torch.randn_like(self.initial_dof_pos[env_ids])
        self.dof_vel[env_ids] = self.initial_dof_vel[env_ids]

        self.actions[env_ids] = 0.0
        self.last_actions[env_ids] = 0.0

        self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["linVelX"][0], self.command_ranges["linVelX"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["linVelY"][0], self.command_ranges["linVelY"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["angVelYaw"][0], self.command_ranges["angVelYaw"][1], (len(env_ids), 1), device=self.device).squeeze(1)

        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.root_states.view(-1, 13)), gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        dof_ids = env_ids_int32 * self.num_dof
        self.gym.set_dof_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.dof_state.view(-1, 2)), gymtorch.unwrap_tensor(dof_ids), len(env_ids_int32))

        self._apply_domain_randomization(env_ids)

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0

    def _apply_domain_randomization(self, env_ids):
        dr_cfg = self.cfg["env"]["domainRand"]
        for env_id in env_ids.tolist():
            env_ptr = self.envs[env_id]
            bridge_handle = self.bridge_handles[env_id]
            robot_handle = self.robot_handles[env_id]

            bridge_shapes = self.gym.get_actor_rigid_shape_properties(env_ptr, bridge_handle)
            fr_low, fr_high = dr_cfg["bridgeFrictionRange"]
            fr = np.random.uniform(fr_low, fr_high)
            for p in bridge_shapes:
                p.friction = fr
            self.gym.set_actor_rigid_shape_properties(env_ptr, bridge_handle, bridge_shapes)

            bridge_props = self.gym.get_actor_rigid_body_properties(env_ptr, bridge_handle)
            bm_low, bm_high = dr_cfg["bridgeMassScaleRange"]
            bm = np.random.uniform(bm_low, bm_high)
            for p in bridge_props:
                p.mass *= bm
            self.gym.set_actor_rigid_body_properties(env_ptr, bridge_handle, bridge_props, recomputeInertia=True)

            dof_props = self.gym.get_actor_dof_properties(env_ptr, robot_handle)
            jd_low, jd_high = dr_cfg["jointDampingRange"]
            damp_scale = np.random.uniform(jd_low, jd_high)
            dof_props["damping"][:self.num_actuated_dof] = self.robot_dof_props["damping"][:self.num_actuated_dof] * damp_scale
            self.gym.set_actor_dof_properties(env_ptr, robot_handle, dof_props)

            rb_props = self.gym.get_actor_rigid_body_properties(env_ptr, robot_handle)
            rm_low, rm_high = dr_cfg["baseMassScaleRange"]
            rm = np.random.uniform(rm_low, rm_high)
            rb_props[self.robot_base_idx].mass *= rm
            self.gym.set_actor_rigid_body_properties(env_ptr, robot_handle, rb_props, recomputeInertia=True)

    def pre_physics_step(self, actions):
        self.actions = actions.clone().to(self.device)
        targets = self.default_dof_pos_tensor.unsqueeze(0) + self.action_scale * self.actions
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(targets))

    def post_physics_step(self):
        self.progress_buf += 1

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.compute_observations()
        self.compute_reward()

        self.last_actions[:] = self.actions


@torch.jit.script
def get_euler_xyz(q):
    qx, qy, qz, qw = 0, 1, 2, 3

    sinr_cosp = 2.0 * (q[:, qw] * q[:, qx] + q[:, qy] * q[:, qz])
    cosr_cosp = q[:, qw] * q[:, qw] - q[:, qx] * q[:, qx] - q[:, qy] * q[:, qy] + q[:, qz] * q[:, qz]
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (q[:, qw] * q[:, qy] - q[:, qz] * q[:, qx])
    pitch = torch.where(torch.abs(sinp) >= 1, copysign(np.pi / 2.0, sinp), torch.asin(sinp))

    siny_cosp = 2.0 * (q[:, qw] * q[:, qz] + q[:, qx] * q[:, qy])
    cosy_cosp = q[:, qw] * q[:, qw] + q[:, qx] * q[:, qx] - q[:, qy] * q[:, qy] - q[:, qz] * q[:, qz]
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw
