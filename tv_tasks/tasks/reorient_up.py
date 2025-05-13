import numpy as np
import os
import torch
from tv_tasks.utils.torch_jit_utils import *
from tv_tasks.tasks.base.shadow_hand import ShadowHandBase
from isaacgym import gymtorch
from isaacgym import gymapi
import random
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R


class ReorientUp(ShadowHandBase):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        self.table_dims = gymapi.Vec3(1, 1, 0.1)
        self.set_camera(cfg, self.table_dims.z + 0.6)
        self.transition_scale = cfg["env"]["transition_scale"]
        self.orientation_scale = cfg["env"]["orientation_scale"]
        self.vel_reward_scale = cfg["env"]["vel_reward_scale"]
        print(f"vel_reward_scale: {self.vel_reward_scale}")
        self.dis_reward_scale = cfg["env"]["dis_reward_scale"]
        print(f"dis_reward_scale: {self.dis_reward_scale}")
        self.ignore_z_dist = cfg["env"]["ignore_z_dist"]
        print(f"ignore_z_dist: {self.ignore_z_dist}")
        self.max_z_theta = cfg["env"].get("max_z_theta", None)
        print(f"max_z_theta_degree: {self.max_z_theta}")
        self.max_z_theta = np.cos(self.max_z_theta * np.pi / 180)
        self.fingertip_tac = cfg["env"]["fingertip_tac"]
        print(f"fingertip_tac --> {self.fingertip_tac}")

        super().__init__(cfg, sim_params, physics_engine, device_type, device_id, headless)
        self.done_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.num_bodies = self.rigid_body_states.view(self.num_envs, -1, 13).shape[1]
        self.rb_forces = torch.zeros((self.num_envs, self.num_bodies, 3), dtype=torch.float, device=self.device)
        self.last_step_rot = torch.zeros(self.num_envs, device=self.device)


    def _create_hand_asset(self):
        # Retrieve asset paths
        self.asset_root = self.cfg["env"]["asset"]["assetRoot"]
        shadow_hand_asset_file = self.cfg["env"]["asset"]["assetFileNameRobot"]

        # load shadow hand_ asset
        self.hand_asset_options = gymapi.AssetOptions()
        self.hand_asset_options.flip_visual_attachments = False
        self.hand_asset_options.fix_base_link = True
        self.hand_asset_options.collapse_fixed_joints = True
        self.hand_asset_options.disable_gravity = True
        self.hand_asset_options.thickness = 0.001
        self.hand_asset_options.angular_damping = 0.01
        self.hand_asset_options.linear_damping = 0.01

        if self.physics_engine == gymapi.SIM_PHYSX:
            self.hand_asset_options.use_physx_armature = True
        # Note - DOF mode is set in the MJCF file and loaded by Isaac Gym
        self.hand_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE

        self.shadow_hand_asset = self.gym.load_asset(self.sim, self.asset_root, shadow_hand_asset_file,
                                                     self.hand_asset_options)

        self.num_shadow_hand_bodies = self.gym.get_asset_rigid_body_count(self.shadow_hand_asset)
        self.num_shadow_hand_shapes = self.gym.get_asset_rigid_shape_count(self.shadow_hand_asset)
        self.num_shadow_hand_dofs = self.gym.get_asset_dof_count(self.shadow_hand_asset)
        self.num_shadow_hand_actuators = self.gym.get_asset_actuator_count(self.shadow_hand_asset)
        self.num_shadow_hand_tendons = self.gym.get_asset_tendon_count(self.shadow_hand_asset)

        print("self.num_shadow_hand_bodies: ", self.num_shadow_hand_bodies)
        print("self.num_shadow_hand_shapes: ", self.num_shadow_hand_shapes)
        print("self.num_shadow_hand_dofs: ", self.num_shadow_hand_dofs)
        print("self.num_shadow_hand_actuators: ", self.num_shadow_hand_actuators)
        print("self.num_shadow_hand_tendons: ", self.num_shadow_hand_tendons)

        # tendon set up
        limit_stiffness = 30
        t_damping = 0.1
        relevant_tendons = ["robot0:T_FFJ1c", "robot0:T_MFJ1c", "robot0:T_RFJ1c", "robot0:T_LFJ1c"]
        tendon_props = self.gym.get_asset_tendon_properties(self.shadow_hand_asset)

        for i in range(self.num_shadow_hand_tendons):
            for rt in relevant_tendons:
                if self.gym.get_asset_tendon_name(self.shadow_hand_asset, i) == rt:
                    tendon_props[i].limit_stiffness = limit_stiffness
                    tendon_props[i].damping = t_damping
        self.gym.set_asset_tendon_properties(self.shadow_hand_asset, tendon_props)

        actuated_dof_names = [self.gym.get_asset_actuator_joint_name(self.shadow_hand_asset, i) for i in
                              range(self.num_shadow_hand_actuators)]
        self.actuated_dof_indices = [self.gym.find_asset_dof_index(self.shadow_hand_asset, name) for name in
                                     actuated_dof_names]
        self.fingertip_handles = [self.gym.find_asset_rigid_body_index(self.shadow_hand_asset, name) for name in
                                  self.fingertips]
        self.sensor_indices = [self.gym.find_asset_rigid_body_index(self.shadow_hand_asset, name) for name in
                               self.force_sensor_body]
        self.shadow_hand_rb_count = self.gym.get_asset_rigid_body_count(self.shadow_hand_asset)

        # get shadow_hand dof properties, loaded by Isaac Gym from the MJCF file
        self.shadow_hand_dof_props = self.gym.get_asset_dof_properties(self.shadow_hand_asset)

        self.shadow_hand_dof_lower_limits = []
        self.shadow_hand_dof_upper_limits = []
        self.shadow_hand_dof_default_pos = []
        self.shadow_hand_dof_default_vel = []

        for i in range(self.num_shadow_hand_dofs):
            self.shadow_hand_dof_lower_limits.append(self.shadow_hand_dof_props['lower'][i])
            self.shadow_hand_dof_upper_limits.append(self.shadow_hand_dof_props['upper'][i])
            self.shadow_hand_dof_default_pos.append(0.0)
            self.shadow_hand_dof_default_vel.append(0.0)

        self.actuated_dof_indices = to_torch(self.actuated_dof_indices, dtype=torch.long, device=self.device)
        self.shadow_hand_dof_lower_limits = to_torch(self.shadow_hand_dof_lower_limits, device=self.device)
        self.shadow_hand_dof_upper_limits = to_torch(self.shadow_hand_dof_upper_limits, device=self.device)
        self.shadow_hand_dof_default_pos = to_torch(self.shadow_hand_dof_default_pos, device=self.device)
        self.shadow_hand_dof_default_vel = to_torch(self.shadow_hand_dof_default_vel, device=self.device)

        # create fingertip force sensors, if needed
        sensor_pose = gymapi.Transform()
        for ft_handle in self.fingertip_handles:
            self.gym.create_asset_force_sensor(self.shadow_hand_asset, ft_handle, sensor_pose)

    def _create_obj_asset(self):
        self.obj_asset_root = self.asset_root + self.cfg["env"]["asset"]["assetFileNameObj"]
        self.env_dict = self.cfg['env']['env_dict']
        # Retrieve asset paths

        self.object_idx = []
        self.num_object_bodies_list = []
        self.num_object_shapes_list = []

        self.object_init_height_dict = {}
        self.hand_init_height_dict = {}
        self.object_asset_dict = {}
        self.goal_asset_dict = {}

        for object_id, object_code in enumerate(self.env_dict):
            # load manipulated object and goal assets
            object_asset_options = gymapi.AssetOptions()
            object_asset_options.convex_decomposition_from_submeshes = True
            # object_asset_options.fix_base_link = False
            # object_asset_options.use_mesh_materials = True
            # object_asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
            # object_asset_options.override_com = True
            # object_asset_options.override_inertia = True

            object_code = str(object_code)
            object_asset_file = "coacd_1.urdf"
            object_asset = self.gym.load_asset(self.sim, self.obj_asset_root + object_code + "/coacd",
                                               object_asset_file, object_asset_options)

            object_asset_options.disable_gravity = True
            goal_asset = self.gym.load_asset(self.sim, self.obj_asset_root + object_code + "/coacd",
                                             object_asset_file, object_asset_options)

            self.object_asset_dict[object_id] = object_asset
            self.goal_asset_dict[object_id] = goal_asset

            self.object_idx.append(object_id)

            self.num_object_bodies_list.append(self.gym.get_asset_rigid_body_count(object_asset))
            self.num_object_shapes_list.append(self.gym.get_asset_rigid_shape_count(object_asset))

            # set object dof properties
            self.num_object_dofs = self.gym.get_asset_dof_count(object_asset)
            object_dof_props = self.gym.get_asset_dof_properties(object_asset)

            self.object_dof_lower_limits = []
            self.object_dof_upper_limits = []

            self.scale = 0.04 # changed from 0.07 to 0.05 and then 0.04

            for i in range(self.num_object_dofs):
                self.object_dof_lower_limits.append(object_dof_props['lower'][i])
                self.object_dof_upper_limits.append(object_dof_props['upper'][i])

            self.object_dof_lower_limits = to_torch(self.object_dof_lower_limits, device=self.device)
            self.object_dof_upper_limits = to_torch(self.object_dof_upper_limits, device=self.device)

            # self.num_object_dofs += 2

    def _create_envs(self, num_envs, spacing, num_per_row):

        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        # load assets
        self._create_hand_asset()
        self._create_obj_asset()
        # self._create_table_asset(self.table_dims)

        self.envs_config()
        self.arm_indices = []

        self.object_idx = to_torch(self.object_idx, dtype=torch.int32, device=self.device)
        object_idx_list = [idx.item() for idx in self.object_idx]
        self.obj_actors = []
        self.env_rigid_count = []
        self.goal_init_state = []

        # set start pos
        shadow_hand_start_pose = gymapi.Transform()
        shadow_hand_start_pose.p = gymapi.Vec3(*get_axis_params(0.5, self.up_axis_idx))


        object_start_pose = gymapi.Transform()
        object_start_pose.p = gymapi.Vec3()
        object_start_pose.r = gymapi.Quat().from_euler_zyx(0, 0, 0)
        object_start_pose.p.x = shadow_hand_start_pose.p.x
        pose_dy, pose_dz = -0.39, 0.06

        object_start_pose.p.y = shadow_hand_start_pose.p.y + pose_dy
        object_start_pose.p.z = shadow_hand_start_pose.p.z + pose_dz

        self.goal_displacement = gymapi.Vec3(-0.2, -0.06, 0.12)
        self.goal_displacement = gymapi.Vec3(0, 0, -0.01)


        self.goal_displacement_tensor = to_torch(
            [self.goal_displacement.x, self.goal_displacement.y, self.goal_displacement.z], device=self.device)
        goal_start_pose = gymapi.Transform()
        goal_start_pose.p = object_start_pose.p + self.goal_displacement

        for i in range(self.num_envs):
            object_idx_this_env = i % len(object_idx_list)
            self.obj_actors.append([])

            # create env instance
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)
            self.env_rigid_count.append(self.gym.get_env_rigid_body_count(env_ptr))

            # compute aggregate size
            max_agg_bodies = self.num_shadow_hand_bodies + 2 * self.num_object_bodies_list[object_idx_this_env] + 1
            max_agg_shapes = self.num_shadow_hand_shapes + 2 * self.num_object_shapes_list[object_idx_this_env] + 1
            if self.aggregate_mode >= 1:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # add hand
            shadow_hand_actor = self._load_shadow_hand(env_ptr, i, self.shadow_hand_asset,
                                                       self.shadow_hand_dof_props,
                                                       shadow_hand_start_pose)
            self.hand_start_states.append(
                [shadow_hand_start_pose.p.x, shadow_hand_start_pose.p.y, shadow_hand_start_pose.p.z,
                 shadow_hand_start_pose.r.x, shadow_hand_start_pose.r.y, shadow_hand_start_pose.r.z,
                 shadow_hand_start_pose.r.w,
                 0, 0, 0, 0, 0, 0])
            self.shadow_hands.append(shadow_hand_actor)

            # add object
            object_actor = self._load_object(env_ptr, i, self.object_asset_dict[object_idx_this_env], object_start_pose,
                                             self.scale)
            self.object_init_state.append([object_start_pose.p.x, object_start_pose.p.y, object_start_pose.p.z,
                                           object_start_pose.r.x, object_start_pose.r.y, object_start_pose.r.z,
                                           object_start_pose.r.w,
                                           0, 0, 0, 0, 0, 0])

            # set friction
            object_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, object_actor)
            for object_shape_prop in object_shape_props:
                object_shape_prop.friction = 0.8
            self.gym.set_actor_rigid_shape_properties(env_ptr, object_actor, object_shape_props)

            # set mass
            # object_body_props = self.gym.get_actor_rigid_body_properties(env_ptr, object_actor)
            # object_body_props[0].mass = 0.15
            # object_body_props[0].mass = 0.05
            # self.gym.set_actor_rigid_body_properties(env_ptr, object_actor, object_body_props)

            self.obj_actors[i].append(object_actor)

            # add goal object
            goal_actor = self._load_goal(env_ptr, i, self.goal_asset_dict[object_idx_this_env], goal_start_pose,
                                         scale=0.001)
            self.goal_init_state.append([goal_start_pose.p.x, goal_start_pose.p.y, goal_start_pose.p.z,
                                         goal_start_pose.r.x, goal_start_pose.r.y, goal_start_pose.r.z,
                                         object_start_pose.r.w,
                                         0, 0, 0, 0, 0, 0])

            # set color
            colorx = 204/255 # random.uniform(0, 1)
            colory = 204/255 # random.uniform(0, 1)
            colorz = 0 # random.uniform(0, 1)
            obj_color = gymapi.Vec3(colorx, colory, colorz)


            # obj_color = gymapi.Vec3(242 / 255, 193 / 255, 46 / 255)
            for o in range(self.num_object_bodies_list[i % len(self.env_dict)]):
                self.gym.set_rigid_body_color(env_ptr, object_actor, o, gymapi.MESH_VISUAL, obj_color)
                self.gym.set_rigid_body_color(env_ptr, goal_actor, o, gymapi.MESH_VISUAL, obj_color)


            # Vision
            if self.cfg["env"]["obs_type"] in ["VisTac", "VisOnly"]:
                self._load_cameras(env_ptr, i, self.camera_props, self.camera_eye_list, self.camera_lookat_list)

            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            self.envs.append(env_ptr)

        self.object_init_state = to_torch(self.object_init_state, device=self.device, dtype=torch.float).view(
            self.num_envs, 13)
        self.init_object_rot = self.object_init_state[:, 3:7].clone()

        self.goal_init_state = to_torch(self.goal_init_state, device=self.device, dtype=torch.float).view(self.num_envs, 13)
        self.goal_states = self.goal_init_state.clone()
        self.goal_states[:, self.up_axis_idx] -= 0.0
        self.goal_states[:, 1] += 0.01

        self.goal_init_state = self.goal_states.clone()
        self.hand_start_states = to_torch(self.hand_start_states, device=self.device).view(self.num_envs, 13)

        self.fingertip_handles = to_torch(self.fingertip_handles, dtype=torch.long, device=self.device)

        self.hand_indices = to_torch(self.hand_indices, dtype=torch.long, device=self.device)
        self.object_indices = to_torch(self.object_indices, dtype=torch.long, device=self.device)
        self.goal_object_indices = to_torch(self.goal_object_indices, dtype=torch.long, device=self.device)

    def compute_reward(self, actions):
        self.rew_buf[:], self.reset_buf[:], self.reset_goal_buf[:], self.progress_buf[:], self.successes[:], self.consecutive_successes[:] = \
            self.compute_hand_reward(self.rew_buf, self.reset_buf, self.reset_goal_buf, self.progress_buf, self.successes,
            self.consecutive_successes,self.max_episode_length, self.object_pos, self.object_rot, self.goal_pos, self.goal_rot,
            self.dist_reward_scale, self.rot_reward_scale, self.rot_eps, self.actions, self.action_penalty_scale,
            self.success_tolerance, self.reach_goal_bonus, self.fall_dist, self.fall_penalty,
            self.max_consecutive_successes, self.av_factor, (self.object_type == "pen")
        )
        idx = torch.where(self.reset_buf==1)[0]
        self.env_mean_successes[idx] = torch.hstack([self.env_mean_successes[idx, 1:], self.successes[idx].view(-1, 1)])
        self.extras['successes'] = self.successes
        self.extras['env_mean_successes'] = self.env_mean_successes.mean(dim=-1)
        self.extras['consecutive_successes'] = self.consecutive_successes


        if self.print_success_stat:
            self.total_resets = self.total_resets + self.reset_buf.sum()
            direct_average_successes = self.total_successes + self.successes.sum()
            self.total_successes = self.total_successes + (self.successes * self.reset_buf).sum()

            # The direct average shows the overall result more quickly, but slightly undershoots long term
            # policy performance.
            print("Direct average consecutive successes = {:.1f}".format(
                direct_average_successes / (self.total_resets + self.num_envs)))
            if self.total_resets > 0:
                print("Post-Reset average consecutive successes = {:.1f}".format(
                    self.total_successes / self.total_resets))

    def compute_hand_reward(
            self, rew_buf, reset_buf, reset_goal_buf, progress_buf, successes, consecutive_successes,
            max_episode_length: float, object_pos, object_rot, target_pos, target_rot,
            dist_reward_scale: float, rot_reward_scale: float, rot_eps: float,
            actions, action_penalty_scale: float,
            success_tolerance: float, reach_goal_bonus: float, fall_dist: float,
            fall_penalty: float, max_consecutive_successes: int, av_factor: float, ignore_z_rot: bool
    ):
        if not self.ignore_z_dist:
            goal_dist = torch.norm(object_pos - target_pos, p=2, dim=-1)
        else:
            goal_dist = torch.norm(object_pos[..., :2] - target_pos[..., :2], p=2, dim=-1)
        goal_z_dist = torch.norm(object_pos[..., 2:] - target_pos[..., 2:], p=2, dim=-1)
        if ignore_z_rot:
            success_tolerance = 2.0 * success_tolerance

        # Orientation alignment for the cube in hand and goal cube
        quat_diff1 = quat_mul(object_rot, quat_conjugate(self.init_object_rot))
        z_rotation = torch.tensor(R.from_quat(quat_diff1.cpu()).as_euler('XYZ', degrees=False), dtype=torch.float32, device=self.device)[:, 2]
        # print(R.from_quat(quat_diff1.cpu()).as_euler('XYZ', degrees=True)[:, 2])

        # rot_dist = 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 0:3], p=2, dim=-1), max=1.0))

        dist_rew = goal_dist * dist_reward_scale
        # rot_rew = 1.0 / (torch.abs(rot_dist) + rot_eps) * rot_reward_scale

        action_penalty = torch.sum(actions ** 2, dim=-1)
        r_vel = torch.clamp(self.object_angvel[:, 2], -10.0, 10.0) * self.vel_reward_scale
        fingertips = torch.stack(
            [self.right_hand_ff_pos, self.right_hand_mf_pos, self.right_hand_rf_pos, self.right_hand_lf_pos,
             self.right_hand_th_pos], dim=1)
        fingertips_z = fingertips[..., -1]  # table height 0.6
        target_dim_z = self.object_pos[..., -1].unsqueeze(-1).repeat(1, 5)
        dis_z = torch.norm(target_dim_z - fingertips_z, p=1, dim=1)
        r_dis = torch.exp(-10 * dis_z) * self.dis_reward_scale  # 0~1
        # Total reward is: position distance + orientation alignment + action regularization + success bonus + fall penalty
        reward = dist_rew + torch.abs(z_rotation) + action_penalty * action_penalty_scale + r_dis + r_vel

        # successes = torch.where(torch.abs(rot_dist) <= success_tolerance, torch.ones_like(successes), successes)
        successes = torch.where(torch.abs(z_rotation) > 3, torch.ones_like(successes), successes)

        # Find out which envs hit the goal and update successes count
        goal_resets = torch.where(successes == 1, torch.ones_like(reset_goal_buf), reset_goal_buf)

        # Success bonus: orientation is within `success_tolerance` of goal orientation
        reward = torch.where(successes == 1, reward + reach_goal_bonus, reward)

        # Fall penalty: distance to the goal is larger than a threshold
        reward = torch.where(goal_z_dist >= fall_dist, reward + fall_penalty, reward)
        # Check env termination conditions, including maximum success number
        resets = torch.where(goal_z_dist >= fall_dist, torch.ones_like(reset_buf), reset_buf)
        # resets = torch.where(torch.abs(rot_dist) <= success_tolerance, torch.ones_like(resets), resets)
        resets = torch.where(successes == 1, torch.ones_like(resets), resets)
        resets = torch.where(goal_dist > 0.04, torch.ones_like(resets), resets)


        if self.max_z_theta is not None:
            after_rot_z = quat_apply(self.object_rot, to_torch([0, 0, 1], device=self.device).unsqueeze(0).repeat(
                self.object_rot.shape[0], 1))
            cos_theta = torch.mul(after_rot_z, to_torch([0, 0, 1], device=self.device)).sum(-1)
            resets = torch.where(cos_theta <= self.max_z_theta, torch.ones_like(resets), resets)
            # reward = torch.where(cos_theta > self.max_z_theta, reward + rot_angle, reward)
            # print("z_theta_reset: " + str(torch.where(cos_theta <= self.max_z_theta)))
        # if max_consecutive_successes > 0:
        #     # Reset progress buffer on goal envs if max_consecutive_successes > 0
        #     progress_buf = torch.where(torch.abs(rot_dist) <= success_tolerance, torch.zeros_like(progress_buf),
        #                                progress_buf)
        #     resets = torch.where(successes >= max_consecutive_successes, torch.ones_like(resets), resets)
        resets = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(resets), resets)

        # Apply penalty for not reaching the goal
        if max_consecutive_successes > 0:
            reward = torch.where(progress_buf >= max_episode_length - 1, reward + 0.5 * fall_penalty, reward)

        num_resets = torch.sum(resets)
        finished_cons_successes = torch.sum(successes * resets.float())

        cons_successes = torch.where(num_resets > 0, av_factor * finished_cons_successes / num_resets + (
                    1.0 - av_factor) * consecutive_successes, consecutive_successes)

        return reward, resets, goal_resets, progress_buf, successes, cons_successes

    def compute_observations(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        # if self.obs_type in ['Real_Binary_Tac', 'Real_Tac']:
        self.gym.refresh_force_sensor_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)

        # robot_state = self.compute_robot_state(full_obs=True)
        # object_state = self.compute_object_state(set_goal=False)
        # base_state = robot_state
        # base_state = torch.cat((robot_state, object_state), dim=1)
        # base_state = torch.clamp(base_state, -self.cfg["env"]["clip_observations"], self.cfg["env"]["clip_observations"])

        self.object_pose = self.root_state_tensor[self.object_indices, 0:7]
        self.object_pos = self.root_state_tensor[self.object_indices, 0:3]
        self.object_rot = self.root_state_tensor[self.object_indices, 3:7]
        self.object_linvel = self.root_state_tensor[self.object_indices, 7:10]
        self.object_angvel = self.root_state_tensor[self.object_indices, 10:13]

        self.goal_pose = self.goal_states[:, 0:7]
        self.goal_pos = self.goal_states[:, 0:3]
        self.goal_rot = self.goal_states[:, 3:7]
        # right hand finger
        self.fingertip_state = self.rigid_body_states[self.fingertip_indices].view(self.num_envs, -1, 13)
        self.fingertip_pos = self.fingertip_state[:, :, 0:3]
        self.fingertip_vel = self.fingertip_state[:, :, 7:13]

        # right hand finger
        self.fingertip_state = self.rigid_body_states[self.fingertip_indices].view(self.num_envs, -1, 13)
        idx = 0
        self.right_hand_ff_pos = self.fingertip_state[:, idx, 0:3]
        self.right_hand_ff_rot = self.fingertip_state[:, idx, 3:7]
        self.right_hand_ff_pos = self.right_hand_ff_pos + quat_apply(self.right_hand_ff_rot,
                                                                     to_torch([0, 0, 1], device=self.device).repeat(
                                                                         self.num_envs, 1) * 0.02)

        idx = 1
        self.right_hand_mf_pos = self.fingertip_state[:, idx, 0:3]
        self.right_hand_mf_rot = self.fingertip_state[:, idx, 3:7]
        self.right_hand_mf_pos = self.right_hand_mf_pos + quat_apply(self.right_hand_mf_rot,
                                                                     to_torch([0, 0, 1], device=self.device).repeat(
                                                                         self.num_envs, 1) * 0.02)

        idx = 2
        self.right_hand_rf_pos = self.fingertip_state[:, idx, 0:3]
        self.right_hand_rf_rot = self.fingertip_state[:, idx, 3:7]
        self.right_hand_rf_pos = self.right_hand_rf_pos + quat_apply(self.right_hand_rf_rot,
                                                                     to_torch([0, 0, 1], device=self.device).repeat(
                                                                         self.num_envs, 1) * 0.02)

        idx = 3
        self.right_hand_lf_pos = self.fingertip_state[:, idx, 0:3]
        self.right_hand_lf_rot = self.fingertip_state[:, idx, 3:7]
        self.right_hand_lf_pos = self.right_hand_lf_pos + quat_apply(self.right_hand_lf_rot,
                                                                     to_torch([0, 0, 1], device=self.device).repeat(
                                                                         self.num_envs, 1) * 0.02)

        idx = 4
        self.right_hand_th_pos = self.fingertip_state[:, idx, 0:3]
        self.right_hand_th_rot = self.fingertip_state[:, idx, 3:7]
        self.right_hand_th_pos = self.right_hand_th_pos + quat_apply(self.right_hand_th_rot,
                                                                     to_torch([0, 0, 1], device=self.device).repeat(
                                                                         self.num_envs, 1) * 0.02)

        base_state = self.compute_robot_state()
        base_state = torch.clamp(base_state, -self.cfg["env"]["clip_observations"],
                                 self.cfg["env"]["clip_observations"])

        touch_force_obs = self.compute_sensor_obs()
        self.touch_force_obs = touch_force_obs

        if self.obs_type == 'VisTac':
            # pixel observation
            pixel_obs = self.compute_pixel_obs()
            # force sensor
            self.obs_states_buf = torch.cat((base_state, pixel_obs, touch_force_obs), dim=1)

        elif self.obs_type == 'TacOnly':
            # force sensor
            # touch_force_obs = self.compute_sensor_obs()
            self.obs_states_buf = torch.cat((base_state, touch_force_obs), dim=1)

        elif self.obs_type == 'VisOnly':
            # pixel observation
            pixel_obs = self.compute_pixel_obs()
            self.obs_states_buf = torch.cat((base_state, pixel_obs), dim=1)

        elif self.obs_type in ['Base', 'BaseSplit','BaseSplit2']:
            self.obs_states_buf = base_state

        elif self.obs_type == 'Real_Tac':
            self.obs_states_buf = torch.cat((base_state, self.force_torque_obs_scale * self.vec_sensor_tensor), dim=1)

        elif self.obs_type == 'Our_Binary_Tac':
            real_binary_tac = self.compute_sensor_obs()
            # self.obs_states_buf = torch.cat((base_state[:, 0:161], real_binary_tac, base_state[:, 191:211]), dim=1)
            self.obs_states_buf = torch.cat((base_state, real_binary_tac), dim=1)

        elif self.obs_type == 'Real_Binary_Tac':
            real_binary_tac = self.compute_fingertip_obs()
            # self.obs_states_buf = torch.cat((base_state[:, 0:161], real_binary_tac, base_state[:, 191:211]), dim=1)
            self.obs_states_buf = torch.cat((base_state, real_binary_tac), dim=1)



    # def compute_sensor_obs(self):
    #     # forces and torques
    #     contact = self.contact_force[self.hand_contact_idx].view(self.num_envs, self.num_force_sensors, 3)
    #     vec_sensor = torch.norm(contact, p=2, dim=2)
    #     fingertip_force = self.vec_sensor_tensor.view(self.num_envs, 5, 6)[:, :, :3]
    #     fingertip_force = torch.norm(fingertip_force, p=2, dim=2)
    #     fingertip_sensor = torch.zeros_like(fingertip_force)
    #     fingertip_sensor[fingertip_force > 0.2] = 1
    #
    #     self.sensor_obs = torch.zeros_like(vec_sensor)
    #     self.sensor_obs[vec_sensor > 0.01] = 1
    #     # self.sensor_obs[:, 0:5] = torch.logical_or(self.sensor_obs[:, 0:5], fingertip_sensor)
    #     # print(self.sensor_obs[:, :5])
    #     return self.sensor_obs
    def compute_sensor_obs(self):
        # forces and torques
        contact = self.contact_force[self.hand_contact_idx].view(self.num_envs, self.num_force_sensors, 3)
        # vec_sensor = self.vec_sensor_tensor
        vec_sensor = contact
        vec_sensor = torch.norm(vec_sensor, p=2, dim=2)
        self.sensor_obs = torch.zeros_like(vec_sensor)
        self.sensor_obs[vec_sensor > 0.01] = 1
        if self.fingertip_tac:
            fingertip_force = self.vec_sensor_tensor.view(self.num_envs, 5, 6)[:, :, :3]
            fingertip_force = torch.norm(fingertip_force, p=2, dim=2)
            fingertip_sensor = torch.zeros_like(fingertip_force)
            fingertip_sensor[fingertip_force > 0.2] = 1
            self.sensor_obs[:, 0:5] = torch.logical_or(self.sensor_obs[:, 0:5], fingertip_sensor)
            # print(self.sensor_obs[:, :5])
        # print(vec_sensor)
        return self.sensor_obs
    def compute_fingertip_obs(self):
        fingertip_force = self.vec_sensor_tensor.view(self.num_envs, 5, 6)[:, :, :3]
        fingertip_force = torch.norm(fingertip_force, p=2, dim=2)
        fingertip_sensor = torch.zeros_like(fingertip_force)
        fingertip_sensor[fingertip_force > 0.2] = 1
        # print(fingertip_sensor)

        return fingertip_sensor

    def compute_robot_state(self, full_obs=False):

        robot_qpos = unscale(self.shadow_hand_dof_pos, self.shadow_hand_dof_lower_limits,
                             self.shadow_hand_dof_upper_limits)
        robot_qves = self.vel_obs_scale * self.shadow_hand_dof_vel
        robot_dof_force = self.force_torque_obs_scale * self.dof_force_tensor
        quat_dist = quat_mul(self.object_rot, quat_conjugate(self.goal_rot))
        self.fingertip_state = self.rigid_body_states[self.fingertip_indices].view(self.num_envs, self.num_fingertips,
                                                                                   13)
        self.fingertip_pos = self.fingertip_state[:, :, 0:3]
        num_ft_states = 13 * self.num_fingertips  # 65
        fingertip_state = self.fingertip_state.reshape(self.num_envs, num_ft_states)
        fingertip_force = self.force_torque_obs_scale * self.vec_sensor_tensor[:, :30]
        # all_robot_state = torch.cat((robot_qpos, robot_qves, robot_dof_force, fingertip_state, fingertip_force,
        #                              self.actions), dim=1)

        if full_obs:
            all_robot_state = torch.cat((robot_qpos,
                                         robot_qves,
                                         robot_dof_force,
                                         self.object_pose,
                                         self.object_linvel,
                                         self.vel_obs_scale * self.object_angvel,
                                         self.goal_pose,
                                         quat_dist,
                                         self.fingertip_state.reshape(self.num_envs, 13 * self.num_fingertips),
                                         # self.force_torque_obs_scale * self.vec_sensor_tensor,
                                         self.actions), dim=1)
        else:
            all_robot_state = torch.cat((robot_qpos,
                                         robot_qves, # 48
                                         # robot_dof_force, # 72
                                         #
                                         # self.object_pose,
                                         # self.object_linvel,
                                         # self.vel_obs_scale * self.object_angvel,
                                         # self.goal_pose,
                                         # quat_dist,  # 24
                                         #
                                         # self.fingertip_state.reshape(self.num_envs, 13 * self.num_fingertips),
                                         # self.actions,

                                         # self.force_torque_obs_scale * self.vec_sensor_tensor,

                                         ), dim=1)  # 65 + 20 = 85
        return all_robot_state

    def compute_expert_state(self):
        return self.compute_robot_state(True)

    # def reset_target_pose(self, env_ids, apply_reset=False):
    #     rand_floats = torch_rand_float(-1.0, 1.0, (len(env_ids), 4), device=self.device)
    #
    #     new_rot = randomize_rotation(rand_floats[:, 0], rand_floats[:, 1], self.x_unit_tensor[env_ids],
    #                                  self.y_unit_tensor[env_ids])
    #
    #     self.goal_states[env_ids, 0:3] = self.goal_init_state[env_ids, 0:3]
    #     self.goal_states[env_ids, 3:7] = new_rot
    #     self.root_state_tensor[self.goal_object_indices[env_ids], 0:3] = self.goal_states[env_ids,
    #                                                                      0:3] + self.goal_displacement_tensor
    #     self.root_state_tensor[self.goal_object_indices[env_ids], 3:7] = self.goal_states[env_ids, 3:7]
    #     self.root_state_tensor[self.goal_object_indices[env_ids], 7:13] = torch.zeros_like(
    #         self.root_state_tensor[self.goal_object_indices[env_ids], 7:13])
    #
    #     if apply_reset:
    #         goal_object_indices = self.goal_object_indices[env_ids].to(torch.int32)
    #         self.gym.set_actor_root_state_tensor_indexed(self.sim,
    #                                                      gymtorch.unwrap_tensor(self.root_state_tensor),
    #                                                      gymtorch.unwrap_tensor(goal_object_indices), len(env_ids))
    #     self.reset_goal_buf[env_ids] = 0

    def reset_target_pose(self, env_ids, apply_reset=False):
        rand_floats = torch_rand_float(-1.0, 1.0, (len(env_ids), 4), device=self.device)

        # new_rot = randomize_rotation(rand_floats[:, 0], rand_floats[:, 1], self.x_unit_tensor[env_ids], self.y_unit_tensor[env_ids])
        target_euler = to_torch([0, 0, 1.57], device=self.device)
        new_rot = quat_from_euler_xyz(target_euler[0], target_euler[1], target_euler[2]).unsqueeze(0).repeat(
            len(env_ids), 1)
        # self.goal_states[env_ids, 0:3] = self.goal_init_state[env_ids, 0:3]
        self.goal_states[env_ids, 0:3] = self.goal_init_state[env_ids, 0:3]
        self.goal_states[env_ids, 3:7] = quat_mul(self.root_state_tensor[self.object_indices[env_ids], 3:7], new_rot)
        self.root_state_tensor[self.goal_object_indices[env_ids], 0:3] = self.goal_states[env_ids,
                                                                         0:3] + self.goal_displacement_tensor
        self.root_state_tensor[self.goal_object_indices[env_ids], 3:7] = self.goal_states[env_ids, 3:7]
        self.root_state_tensor[self.goal_object_indices[env_ids], 7:13] = torch.zeros_like(
            self.root_state_tensor[self.goal_object_indices[env_ids], 7:13])

        if apply_reset:
            goal_object_indices = self.goal_object_indices[env_ids].to(torch.int32)
            self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                         gymtorch.unwrap_tensor(self.root_state_tensor),
                                                         gymtorch.unwrap_tensor(goal_object_indices), len(env_ids))
        self.reset_goal_buf[env_ids] = 0

    def reset_idx(self, env_ids, goal_env_ids):
        # randomization can happen only at reset time, since it can reset actor positions on GPU
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

        # generate random values
        rand_floats = torch_rand_float(-1.0, 1.0, (len(env_ids), self.num_shadow_hand_dofs * 2 + 5), device=self.device)

        self.rb_forces[env_ids, :, :] = 0.0

        # reset object
        self.root_state_tensor[self.object_indices[env_ids]] = self.object_init_state[env_ids].clone()
        self.root_state_tensor[self.object_indices[env_ids], 0:2] = self.object_init_state[env_ids,
                                                                    0:2] + self.reset_position_noise * rand_floats[:,
                                                                                                       0:2]
        # self.root_state_tensor[self.object_indices[env_ids], self.up_axis_idx] = self.object_init_state[env_ids, self.up_axis_idx] + \
        #     self.reset_position_noise * rand_floats[:, self.up_axis_idx]

        # new_object_rot = randomize_rotation(rand_floats[:, 3], rand_floats[:, 4], self.x_unit_tensor[env_ids], self.y_unit_tensor[env_ids])
        new_object_rot = randomize_z_rotation(rand_floats[:, 3], self.z_unit_tensor[env_ids])
        if self.object_type == "pen":
            rand_angle_y = torch.tensor(0.3)
            new_object_rot = randomize_rotation_pen(rand_floats[:, 3], rand_floats[:, 4], rand_angle_y,
                                                    self.x_unit_tensor[env_ids], self.y_unit_tensor[env_ids],
                                                    self.z_unit_tensor[env_ids])

        self.init_object_rot[env_ids] = new_object_rot

        self.root_state_tensor[self.object_indices[env_ids], 3:7] = new_object_rot # init rotation
        # self.root_state_tensor[self.object_indices[env_ids], 3:7] =  torch.zeros_like(
        #     self.root_state_tensor[self.object_indices[env_ids], 3:7])
        #
        self.root_state_tensor[self.object_indices[env_ids], 7:13] = torch.zeros_like(
            self.root_state_tensor[self.object_indices[env_ids], 7:13])

        object_indices = torch.unique(torch.cat([self.object_indices[env_ids],
                                                 self.goal_object_indices[env_ids],
                                                 self.goal_object_indices[goal_env_ids]]).to(torch.int32))
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_state_tensor),
                                                     gymtorch.unwrap_tensor(object_indices), len(object_indices))
        self.reset_target_pose(env_ids)
        # reset random force probabilities
        self.random_force_prob[env_ids] = torch.exp(
            (torch.log(self.force_prob_range[0]) - torch.log(self.force_prob_range[1]))
            * torch.rand(len(env_ids), device=self.device) + torch.log(self.force_prob_range[1]))

        # reset shadow hand
        delta_max = self.shadow_hand_dof_upper_limits - self.shadow_hand_dof_default_pos
        delta_min = self.shadow_hand_dof_lower_limits - self.shadow_hand_dof_default_pos
        rand_delta = delta_min + (delta_max - delta_min) * 0.5 * (rand_floats[:, 5:5 + self.num_shadow_hand_dofs] + 1)

        pos = self.shadow_hand_default_dof_pos + self.reset_dof_pos_noise * rand_delta
        self.shadow_hand_dof_pos[env_ids, :] = pos
        self.shadow_hand_dof_vel[env_ids, :] = self.shadow_hand_dof_default_vel + \
                                               self.reset_dof_vel_noise * rand_floats[:,
                                                                          5 + self.num_shadow_hand_dofs:5 + self.num_shadow_hand_dofs * 2]
        self.prev_targets[env_ids, :self.num_shadow_hand_dofs] = pos
        self.cur_targets[env_ids, :self.num_shadow_hand_dofs] = pos

        hand_indices = self.hand_indices[env_ids].to(torch.int32)
        self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self.prev_targets),
                                                        gymtorch.unwrap_tensor(hand_indices), len(env_ids))

        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(hand_indices), len(env_ids))

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self.successes[env_ids] = 0

    def pre_physics_step(self, actions):
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        goal_env_ids = self.reset_goal_buf.nonzero(as_tuple=False).squeeze(-1)

        # # if only goals need reset, then call set API
        # if len(goal_env_ids) > 0 and len(env_ids) == 0:
        #     self.reset_target_pose(goal_env_ids, apply_reset=True)
        # # if goals need reset in addition to other envs, call set API in reset_idx()
        # elif len(goal_env_ids) > 0:
        #     self.reset_target_pose(goal_env_ids)

        if len(env_ids) > 0:
            self.reset_idx(env_ids, goal_env_ids)

        self.actions = actions.clone().to(self.device)
        if self.use_relative_control:
            targets = self.prev_targets[:,
                      self.actuated_dof_indices] + self.shadow_hand_dof_speed_scale * self.dt * self.actions
            self.cur_targets[:, self.actuated_dof_indices] = tensor_clamp(targets,
                                                                          self.shadow_hand_dof_lower_limits[
                                                                              self.actuated_dof_indices],
                                                                          self.shadow_hand_dof_upper_limits[
                                                                              self.actuated_dof_indices])
        else:
            self.cur_targets[:, self.actuated_dof_indices] = scale(self.actions,
                                                                   self.shadow_hand_dof_lower_limits[
                                                                       self.actuated_dof_indices],
                                                                   self.shadow_hand_dof_upper_limits[
                                                                       self.actuated_dof_indices])
            self.cur_targets[:, self.actuated_dof_indices] = self.act_moving_average * self.cur_targets[:,
                                                                                       self.actuated_dof_indices] + (
                                                                         1.0 - self.act_moving_average) * self.prev_targets[
                                                                                                          :,
                                                                                                          self.actuated_dof_indices]
            self.cur_targets[:, self.actuated_dof_indices] = tensor_clamp(
                self.cur_targets[:, self.actuated_dof_indices],
                self.shadow_hand_dof_lower_limits[self.actuated_dof_indices],
                self.shadow_hand_dof_upper_limits[self.actuated_dof_indices])

        self.prev_targets[:, self.actuated_dof_indices] = self.cur_targets[:, self.actuated_dof_indices]
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.cur_targets))

        if self.force_scale > 0.0:
            self.rb_forces *= torch.pow(self.force_decay, self.dt / self.force_decay_interval)

            # apply new forces
            force_indices = (torch.rand(self.num_envs, device=self.device) < self.random_force_prob).nonzero()
            self.rb_forces[force_indices, self.object_rb_handles, :] = torch.randn(
                self.rb_forces[force_indices, self.object_rb_handles, :].shape,
                device=self.device) * self.object_rb_masses * self.force_scale

            self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(self.rb_forces), None,
                                                    gymapi.LOCAL_SPACE)

    def post_physics_step(self):
        self.progress_buf += 1
        self.randomize_buf += 1

        self.compute_observations()
        self.compute_reward(self.actions)

        if self.viewer and self.debug_viz:
            # draw axes on target object
            self.gym.clear_lines(self.viewer)
            self.gym.refresh_rigid_body_state_tensor(self.sim)

            for i in range(self.num_envs):
                targetx = (self.goal_pos[i] + quat_apply(self.goal_rot[i], to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
                targety = (self.goal_pos[i] + quat_apply(self.goal_rot[i], to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
                targetz = (self.goal_pos[i] + quat_apply(self.goal_rot[i], to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

                p0 = self.goal_pos[i].cpu().numpy() + self.goal_displacement_tensor.cpu().numpy()
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], targetx[0], targetx[1], targetx[2]], [0.85, 0.1, 0.1])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], targety[0], targety[1], targety[2]], [0.1, 0.85, 0.1])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], targetz[0], targetz[1], targetz[2]], [0.1, 0.1, 0.85])

                objectx = (self.object_pos[i] + quat_apply(self.object_rot[i], to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
                objecty = (self.object_pos[i] + quat_apply(self.object_rot[i], to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
                objectz = (self.object_pos[i] + quat_apply(self.object_rot[i], to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

                p0 = self.object_pos[i].cpu().numpy()
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], objectx[0], objectx[1], objectx[2]], [0.85, 0.1, 0.1])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], objecty[0], objecty[1], objecty[2]], [0.1, 0.85, 0.1])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], objectz[0], objectz[1], objectz[2]], [0.1, 0.1, 0.85])

    def angle_between_vectors(self, tensor1, tensor2):

        dot_products = torch.sum(tensor1 * tensor2, dim=1)

        norm_tensor1 = torch.norm(tensor1.float(), dim=1)
        norm_tensor2 = torch.norm(tensor2.float(), dim=1)

        cosine_similarities = dot_products / (norm_tensor1 * norm_tensor2)

        cosine_similarities = torch.clamp(cosine_similarities, -1.0, 1.0)

        angles_rad = torch.acos(cosine_similarities)

        cross = torch.cross(tensor1.float(), tensor2.float(), dim=1)
        direction = torch.sign(torch.sum(cross, dim=1))

        angles = angles_rad * direction
        return angles_rad

    def calculate_rotation_angle_about_z_axis(self, quaternion_series):

        fixed_z_axis = torch.tensor([0.0, 0.0, 1.0], device=self.device)


        q_w, q_x, q_y, q_z = quaternion_series[:, 3], quaternion_series[:, 0], quaternion_series[:,
                                                                               1], quaternion_series[:, 2]
        R = torch.stack([
            1 - 2 * (q_y ** 2 + q_z ** 2), 2 * (q_x * q_y - q_w * q_z), 2 * (q_x * q_z + q_w * q_y),
            2 * (q_x * q_y + q_w * q_z), 1 - 2 * (q_x ** 2 + q_z ** 2), 2 * (q_y * q_z - q_w * q_x),
            2 * (q_x * q_z - q_w * q_y), 2 * (q_y * q_z + q_w * q_x), 1 - 2 * (q_x ** 2 + q_y ** 2)
        ], dim=1).view(-1, 3, 3)


        z_fixed_in_object_frame = torch.matmul(R, fixed_z_axis)


        rotation_angle_rad = torch.atan2(z_fixed_in_object_frame[:, 1], z_fixed_in_object_frame[:, 0])
        rotation_angle_deg = torch.rad2deg(rotation_angle_rad)

        return rotation_angle_rad

    def add_gaussian_noise(self, tensor, mean=0.0, std=0.1):
        """
        Adds Gaussian noise to a tensor.

        Args:
            tensor (torch.Tensor): Input tensor to add noise to.
            mean (float): Mean of the Gaussian noise (default: 0.0).
            std (float): Standard deviation of the Gaussian noise (default: 0.1).

        Returns:
            torch.Tensor: Tensor with Gaussian noise added.
        """
        noise = torch.randn_like(tensor, device=self.device) * std + mean
        # print(noise)
        noisy_tensor = tensor + noise
        return noisy_tensor


#####################################################################
###=========================jit functions=========================###
#####################################################################




@torch.jit.script
def randomize_rotation(rand0, rand1, x_unit_tensor, y_unit_tensor):
    return quat_mul(quat_from_angle_axis(rand0 * np.pi, x_unit_tensor),
                    quat_from_angle_axis(rand1 * np.pi, y_unit_tensor))


@torch.jit.script
def randomize_rotation_pen(rand0, rand1, max_angle, x_unit_tensor, y_unit_tensor, z_unit_tensor):
    rot = quat_mul(quat_from_angle_axis(0.5 * np.pi + rand0 * max_angle, x_unit_tensor),
                   quat_from_angle_axis(rand0 * np.pi, z_unit_tensor))
    return rot

@torch.jit.script
def randomize_z_rotation(rand0, z_unit_tensor):
    return quat_from_angle_axis(rand0 * np.pi, z_unit_tensor)
