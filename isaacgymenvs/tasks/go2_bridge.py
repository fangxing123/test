import math
from typing import Tuple

import torch
from isaacgym import gymapi, gymtorch, torch_utils

from isaacgymenvs.tasks.base.vec_task import VecTask


class Go2Bridge(VecTask):
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.cfg = cfg

        self.lin_vel_scale = self.cfg["env"]["learn"]["linearVelocityScale"]
        self.ang_vel_scale = self.cfg["env"]["learn"]["angularVelocityScale"]
        self.dof_pos_scale = self.cfg["env"]["learn"]["dofPositionScale"]
        self.dof_vel_scale = self.cfg["env"]["learn"]["dofVelocityScale"]
        self.action_scale = self.cfg["env"]["control"]["actionScale"]
        self.decimation = self.cfg["env"]["control"]["decimation"]
        self.dt = self.cfg["sim"]["dt"] * self.decimation

        self.max_episode_length_s = self.cfg["env"]["episodeLength_s"]
        self.max_episode_length = int(self.max_episode_length_s / self.dt + 0.5)

        self.command_x_range = self.cfg["env"]["commands"]["linVelX"]
        self.command_y_range = self.cfg["env"]["commands"]["linVelY"]
        self.command_yaw_range = self.cfg["env"]["commands"]["angVelYaw"]

        self.default_dof_pos = torch.tensor(self.cfg["env"]["defaultJointAngles"], dtype=torch.float)

        self.cfg["env"]["numObservations"] = 69
        self.cfg["env"]["numActions"] = 12

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device,
                         graphics_device_id=graphics_device_id, headless=headless,
                         virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        force_sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)

        self.root_states = gymtorch.wrap_tensor(actor_root_state).view(self.num_envs, -1, 13)
        self.robot_root_states = self.root_states[:, 0, :]

        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]

        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state).view(self.num_envs, self.num_bodies, 13)
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, self.num_bodies, 3)
        self.force_sensor_tensor = gymtorch.wrap_tensor(force_sensor_tensor).view(self.num_envs, 4, 6)

        self.base_quat = self.robot_root_states[:, 3:7]
        self.base_lin_vel = torch_utils.quat_rotate_inverse(self.base_quat, self.robot_root_states[:, 7:10])
        self.base_ang_vel = torch_utils.quat_rotate_inverse(self.base_quat, self.robot_root_states[:, 10:13])
        self.projected_gravity = torch_utils.quat_rotate_inverse(
            self.base_quat,
            self.gravity_vec.repeat((self.num_envs, 1))
        )

        self.commands = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device)
        self.last_actions = torch.zeros_like(self.actions)

        self.default_dof_pos = self.default_dof_pos.to(self.device).unsqueeze(0)

        self.robot_actor_indices = torch.tensor(self.robot_actor_indices, dtype=torch.int32, device=self.device)
        self.bridge_actor_indices = torch.tensor(self.bridge_actor_indices, dtype=torch.int32, device=self.device)

        self.feet_indices = torch.tensor(self.feet_indices, dtype=torch.long, device=self.device)
        self.bridge_link_indices = torch.tensor(self.bridge_link_indices, dtype=torch.long, device=self.device)

        self.initial_root_states = self.root_states.clone()
        self.initial_dof_pos = self.default_dof_pos.repeat(self.num_envs, 1)
        self.initial_dof_vel = torch.zeros_like(self.dof_vel)

        self.extras["episode"] = {}
        self.reset_idx(torch.arange(self.num_envs, device=self.device, dtype=torch.long))

    def create_sim(self):
        self.up_axis_idx = 2
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]["envSpacing"], int(math.sqrt(self.num_envs)))

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg["env"]["plane"]["staticFriction"]
        plane_params.dynamic_friction = self.cfg["env"]["plane"]["dynamicFriction"]
        plane_params.restitution = self.cfg["env"]["plane"]["restitution"]
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = self.cfg["env"]["asset"]["assetRoot"]
        robot_asset_file = self.cfg["env"]["asset"]["robotAssetFile"]
        bridge_asset_file = self.cfg["env"]["asset"]["bridgeAssetFile"]

        robot_opts = gymapi.AssetOptions()
        robot_opts.default_dof_drive_mode = gymapi.DOF_MODE_POS
        robot_opts.replace_cylinder_with_capsule = True
        robot_opts.fix_base_link = False
        robot_opts.collapse_fixed_joints = True
        robot_opts.disable_gravity = False

        bridge_opts = gymapi.AssetOptions()
        bridge_opts.fix_base_link = False
        bridge_opts.collapse_fixed_joints = False
        bridge_opts.disable_gravity = False

        robot_asset = self.gym.load_asset(self.sim, asset_root, robot_asset_file, robot_opts)
        bridge_asset = self.gym.load_asset(self.sim, asset_root, bridge_asset_file, bridge_opts)

        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset) + self.gym.get_asset_rigid_body_count(bridge_asset)

        dof_props = self.gym.get_asset_dof_properties(robot_asset)
        dof_props["driveMode"].fill(gymapi.DOF_MODE_POS)
        dof_props["stiffness"].fill(self.cfg["env"]["control"]["stiffness"])
        dof_props["damping"].fill(self.cfg["env"]["control"]["damping"])

        foot_names = self.cfg["env"]["asset"]["footNames"]
        all_robot_body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.feet_indices = [all_robot_body_names.index(name) for name in foot_names]

        sensor_pose = gymapi.Transform()
        for foot_idx in self.feet_indices:
            self.gym.create_asset_force_sensor(robot_asset, foot_idx, sensor_pose)

        bridge_body_names = self.gym.get_asset_rigid_body_names(bridge_asset)
        self.bridge_link_indices = [i + len(all_robot_body_names) for i in range(len(bridge_body_names))]

        robot_start = gymapi.Transform()
        robot_start.p = gymapi.Vec3(-4.0, 0.0, 0.42)
        bridge_start = gymapi.Transform()
        bridge_start.p = gymapi.Vec3(0.0, 0.0, 0.25)

        self.robot_actor_indices = []
        self.bridge_actor_indices = []
        self.envs = []

        for i in range(num_envs):
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)

            robot_handle = self.gym.create_actor(env_ptr, robot_asset, robot_start, "go2", i, 0, 0)
            self.gym.set_actor_dof_properties(env_ptr, robot_handle, dof_props)

            bridge_handle = self.gym.create_actor(env_ptr, bridge_asset, bridge_start, "bridge", i, 0, 0)

            robot_index = self.gym.get_actor_index(env_ptr, robot_handle, gymapi.DOMAIN_SIM)
            bridge_index = self.gym.get_actor_index(env_ptr, bridge_handle, gymapi.DOMAIN_SIM)
            self.robot_actor_indices.append(robot_index)
            self.bridge_actor_indices.append(bridge_index)

            self._setup_bridge_dynamics(env_ptr, bridge_handle)
            self._apply_domain_randomization(env_ptr, robot_handle, bridge_handle)

            self.envs.append(env_ptr)

    def _setup_bridge_dynamics(self, env_ptr, bridge_handle):
        rb_props = self.gym.get_actor_rigid_body_properties(env_ptr, bridge_handle)
        for i, prop in enumerate(rb_props):
            if 0 < i < len(rb_props) - 1:
                prop.linear_damping = self.cfg["env"]["bridge"]["linearDamping"]
                prop.angular_damping = self.cfg["env"]["bridge"]["angularDamping"]
        self.gym.set_actor_rigid_body_properties(env_ptr, bridge_handle, rb_props, recomputeInertia=True)

        dof_props = self.gym.get_actor_dof_properties(env_ptr, bridge_handle)
        if len(dof_props) > 0:
            for i in range(len(dof_props)):
                dof_props["damping"][i] = self.cfg["env"]["bridge"]["jointDamping"]
            self.gym.set_actor_dof_properties(env_ptr, bridge_handle, dof_props)

    def _apply_domain_randomization(self, env_ptr, robot_handle, bridge_handle):
        dr_cfg = self.cfg["env"]["domainRand"]

        bridge_rb_props = self.gym.get_actor_rigid_shape_properties(env_ptr, bridge_handle)
        friction_scale = torch.rand(1).item() * (dr_cfg["bridgeFriction"][1] - dr_cfg["bridgeFriction"][0]) + dr_cfg["bridgeFriction"][0]
        for prop in bridge_rb_props:
            prop.friction = friction_scale
        self.gym.set_actor_rigid_shape_properties(env_ptr, bridge_handle, bridge_rb_props)

        bridge_body_props = self.gym.get_actor_rigid_body_properties(env_ptr, bridge_handle)
        mass_scale = torch.rand(1).item() * (dr_cfg["bridgeMassScale"][1] - dr_cfg["bridgeMassScale"][0]) + dr_cfg["bridgeMassScale"][0]
        for prop in bridge_body_props:
            prop.mass *= mass_scale
        self.gym.set_actor_rigid_body_properties(env_ptr, bridge_handle, bridge_body_props, recomputeInertia=True)

        robot_dof_props = self.gym.get_actor_dof_properties(env_ptr, robot_handle)
        damping_scale = torch.rand(1).item() * (dr_cfg["jointDampingScale"][1] - dr_cfg["jointDampingScale"][0]) + dr_cfg["jointDampingScale"][0]
        robot_dof_props["damping"] *= damping_scale
        self.gym.set_actor_dof_properties(env_ptr, robot_handle, robot_dof_props)

        robot_body_props = self.gym.get_actor_rigid_body_properties(env_ptr, robot_handle)
        base_mass_scale = torch.rand(1).item() * (dr_cfg["baseMassScale"][1] - dr_cfg["baseMassScale"][0]) + dr_cfg["baseMassScale"][0]
        robot_body_props[0].mass *= base_mass_scale
        self.gym.set_actor_rigid_body_properties(env_ptr, robot_handle, robot_body_props, recomputeInertia=True)

    def compute_observations(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.base_quat = self.robot_root_states[:, 3:7]
        self.base_lin_vel = torch_utils.quat_rotate_inverse(self.base_quat, self.robot_root_states[:, 7:10])
        self.base_ang_vel = torch_utils.quat_rotate_inverse(self.base_quat, self.robot_root_states[:, 10:13])
        self.projected_gravity = torch_utils.quat_rotate_inverse(self.base_quat, self.gravity_vec.repeat((self.num_envs, 1)))

        feet_forces = self.force_sensor_tensor[:, :, :3].reshape(self.num_envs, -1)

        bridge_summary = []
        bridge_local_pos = self.rigid_body_states[:, self.bridge_link_indices, 0:3]
        robot_pos = self.robot_root_states[:, 0:3].unsqueeze(1)
        dist = torch.norm(bridge_local_pos - robot_pos, dim=-1)
        closest_idx = torch.topk(dist, k=4, largest=False).indices

        bridge_quat = self.rigid_body_states[:, self.bridge_link_indices, 3:7]
        for i in range(4):
            q = bridge_quat[torch.arange(self.num_envs, device=self.device), closest_idx[:, i]]
            roll, pitch, _ = torch_utils.get_euler_xyz(q)
            bridge_summary.append(torch.stack([roll, pitch], dim=-1))
        bridge_summary = torch.cat(bridge_summary, dim=-1)

        self.obs_buf = torch.cat((
            self.base_lin_vel * self.lin_vel_scale,
            self.base_ang_vel * self.ang_vel_scale,
            self.projected_gravity,
            (self.dof_pos - self.default_dof_pos) * self.dof_pos_scale,
            self.dof_vel * self.dof_vel_scale,
            self.last_actions,
            self.commands,
            feet_forces,
            bridge_summary
        ), dim=-1)

        return self.obs_buf

    def compute_reward(self):
        feet_vel = self.rigid_body_states[:, self.feet_indices, 7:10]
        feet_contacts = torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) > 1.0

        rewards, reset = compute_go2_bridge_reward(
            self.base_lin_vel,
            self.base_ang_vel,
            self.projected_gravity,
            self.commands,
            self.actions,
            self.last_actions,
            self.robot_root_states[:, 0:3],
            feet_vel,
            feet_contacts,
            self.progress_buf,
            self.max_episode_length,
            self.cfg["env"]["rewardScales"]
        )

        self.rew_buf[:] = rewards
        self.reset_buf[:] = reset

    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return

        self.dof_pos[env_ids] = self.initial_dof_pos[env_ids]
        self.dof_vel[env_ids] = self.initial_dof_vel[env_ids]
        self.actions[env_ids] = 0.0
        self.last_actions[env_ids] = 0.0

        self.robot_root_states[env_ids] = self.initial_root_states[env_ids, 0, :]
        self.robot_root_states[env_ids, 0] += torch_rand_float(-0.2, 0.2, (len(env_ids), 1), device=self.device).squeeze(-1)
        self.robot_root_states[env_ids, 1] += torch_rand_float(-0.1, 0.1, (len(env_ids), 1), device=self.device).squeeze(-1)

        self.commands[env_ids, 0] = torch_rand_float(self.command_x_range[0], self.command_x_range[1], (len(env_ids), 1), device=self.device).squeeze(-1)
        self.commands[env_ids, 1] = torch_rand_float(self.command_y_range[0], self.command_y_range[1], (len(env_ids), 1), device=self.device).squeeze(-1)
        self.commands[env_ids, 2] = torch_rand_float(self.command_yaw_range[0], self.command_yaw_range[1], (len(env_ids), 1), device=self.device).squeeze(-1)

        actor_indices = self.robot_actor_indices[env_ids].to(torch.int32)

        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_states.view(-1, 13)),
            gymtorch.unwrap_tensor(actor_indices),
            len(actor_indices)
        )
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(actor_indices),
            len(actor_indices)
        )

        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

    def pre_physics_step(self, actions):
        self.actions = torch.clamp(actions, -1.0, 1.0)
        targets = self.default_dof_pos + self.actions * self.action_scale

        for _ in range(self.decimation):
            self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(targets))
            self.gym.simulate(self.sim)
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)

    def post_physics_step(self):
        self.progress_buf += 1

        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.compute_observations()
        self.compute_reward()

        self.last_actions[:] = self.actions[:]


@torch.jit.script
def compute_go2_bridge_reward(
    base_lin_vel,
    base_ang_vel,
    projected_gravity,
    commands,
    actions,
    last_actions,
    base_pos,
    feet_vel,
    feet_contacts,
    progress_buf,
    max_episode_length: int,
    reward_scales
) -> Tuple[torch.Tensor, torch.Tensor]:
    lin_vel_error = torch.sum(torch.square(commands[:, :2] - base_lin_vel[:, :2]), dim=1)
    yaw_vel_error = torch.square(commands[:, 2] - base_ang_vel[:, 2])
    tracking_reward = torch.exp(-lin_vel_error * 2.0) + torch.exp(-yaw_vel_error * 1.0)

    forward_progress = base_pos[:, 0]
    on_bridge = ((base_pos[:, 0] > -5.0) & (base_pos[:, 0] < 5.0) & (torch.abs(base_pos[:, 1]) < 0.7)).float()
    upright = torch.square(projected_gravity[:, 2])

    action_smoothness = torch.sum(torch.square(actions - last_actions), dim=1)

    slip = torch.sum(torch.norm(feet_vel[:, :, :2], dim=-1) * feet_contacts.float(), dim=1)

    fallen = (projected_gravity[:, 2] < 0.3) | (base_pos[:, 2] < 0.18)

    reward = (
        reward_scales["tracking"] * tracking_reward
        + reward_scales["forward"] * forward_progress
        + reward_scales["onBridge"] * on_bridge
        + reward_scales["upright"] * upright
        - reward_scales["actionSmoothness"] * action_smoothness
        - reward_scales["footSlip"] * slip
        - reward_scales["fall"] * fallen.float()
    )

    reset = torch.where((progress_buf >= max_episode_length - 1) | fallen, torch.ones_like(progress_buf), torch.zeros_like(progress_buf))
    return reward, reset


def torch_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(*shape, device=device) + lower
