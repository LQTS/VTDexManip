import numpy as np
import os
import torch
from tv_tasks.utils.torch_jit_utils import *
from tv_tasks.tasks.base.shadow_hand import ShadowHandBase
from isaacgym import gymtorch
from isaacgym import gymapi
import random
import copy


class Sliding(ShadowHandBase):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        self.table_dims = gymapi.Vec3(1, 1, 0.6)
        self.set_camera(cfg, self.table_dims.z)
        self.transition_scale = cfg["env"]["transition_scale"]
        self.orientation_scale = cfg["env"]["orientation_scale"]
        super().__init__(cfg, sim_params, physics_engine, device_type, device_id, headless)
        # self.rigid_body_states= self.rigid_body_states.view(self.num_envs, -1, 13)
        self.num_bodies = self.rigid_body_states.view(self.num_envs, -1, 13).shape[1]
        self.apply_forces = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device, dtype=torch.float)
        self.apply_torque = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device, dtype=torch.float)

        self.init_slide_pos = copy.deepcopy(self.rigid_body_states.view(self.num_envs, -1, 13)[:, 46 + 1, 0:3])
        self.pen_right_handle_pos = copy.deepcopy(self.rigid_body_states.view(self.num_envs, -1, 13)[:, 46 + 1, 0:3])
        self.pen_right_handle_rot = copy.deepcopy(self.rigid_body_states.view(self.num_envs, -1, 13)[:, 46 + 1, 3:7])
        self.pen_right_handle_pos[:, 1] += 0.08
        # self.placeholder = torch.zeros([3, 1024, 1024, 1024], dtype=torch.float).cuda(device_id)  #2*4GB GPU MEMORY

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
        # self.hand_asset_options.linear_damping = 10

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

        object_scale = self.cfg["env"]["object_scale"]

        self.object_idx = []
        self.num_object_bodies_list = []
        self.num_object_shapes_list = []

        self.object_init_height_dict = {}
        self.hand_init_height_dict = {}
        self.object_asset_dict = {}
        self.goal_asset_dict = {}
        self.object_scale = {}

        for object_id, object_code in enumerate(self.env_dict):
            # load manipulated object and goal assets
            # load manipulated object and goal assets
            object_asset_options = gymapi.AssetOptions()
            object_asset_options.density = 500
            object_asset_options.fix_base_link = True
            # object_asset_options.collapse_fixed_joints = True
            object_asset_options.disable_gravity = False
            object_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE

            object_code = str(object_code)
            object_asset_file = object_code + ".urdf"
            object_asset = self.gym.load_asset(self.sim, self.obj_asset_root + object_code,
                                               object_asset_file, object_asset_options)

            object_asset_options.disable_gravity = True
            goal_asset = self.gym.load_asset(self.sim, self.obj_asset_root + object_code,
                                             object_asset_file, object_asset_options)

            self.object_asset_dict[object_id] = object_asset
            self.goal_asset_dict[object_id] = goal_asset
            self.object_scale[object_id] = object_scale[object_id]

            self.object_idx.append(object_id)

            self.num_object_bodies_list.append(self.gym.get_asset_rigid_body_count(object_asset))
            self.num_object_shapes_list.append(self.gym.get_asset_rigid_shape_count(object_asset))

            # set object dof properties
            self.num_object_dofs = self.gym.get_asset_dof_count(object_asset)
            object_dof_props = self.gym.get_asset_dof_properties(object_asset)

            self.object_dof_lower_limits = []
            self.object_dof_upper_limits = []

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
        self._create_table_asset(self.table_dims)

        self.envs_config()
        self.arm_indices = []

        self.object_idx = to_torch(self.object_idx, dtype=torch.int32, device=self.device)
        object_idx_list = [idx.item() for idx in self.object_idx]
        self.obj_actors = []
        self.env_rigid_count = []

        for i in range(self.num_envs):
            object_idx_this_env = i % len(object_idx_list)
            self.obj_actors.append([])

            # create env instance
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)
            self.env_rigid_count.append(self.gym.get_env_rigid_body_count(env_ptr))

            # compute aggregate size
            max_agg_bodies = self.num_shadow_hand_bodies + self.num_object_bodies_list[object_idx_this_env] + 1
            max_agg_shapes = self.num_shadow_hand_shapes + self.num_object_shapes_list[object_idx_this_env] + 1
            if self.aggregate_mode >= 1:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

                # set start pos
                shadow_hand_start_pose = gymapi.Transform()
                shadow_hand_start_pose.p = gymapi.Vec3(0.38, 0.12, 0.68)
                shadow_hand_start_pose.r = gymapi.Quat().from_euler_zyx(3.14159, 0, 1.57)

                object_start_pose = gymapi.Transform()
                object_start_pose.p = gymapi.Vec3(0.0, 0., 0.63)
                # object_start_pose.r = gymapi.Quat().from_euler_zyx(1.57, 1.57, 0)

                if object_idx_this_env in [0]:
                    shadow_hand_start_pose.p = gymapi.Vec3(0.4, 0.1, 0.71)
                    object_start_pose.r = gymapi.Quat().from_euler_zyx(1.57, 0, 0)
                elif object_idx_this_env in [1]:
                    shadow_hand_start_pose.p = gymapi.Vec3(0.4, 0.08, 0.71)
                    object_start_pose.p = gymapi.Vec3(0.0, 0, 0.67)
                    object_start_pose.r = gymapi.Quat().from_euler_zyx(-1.57, 0, -1.57)
                elif object_idx_this_env in [2]:
                    shadow_hand_start_pose.p = gymapi.Vec3(0.4, 0.13, 0.71)
                    object_start_pose.p = gymapi.Vec3(0.0, -0.12, 0.65)
                    object_start_pose.r = gymapi.Quat().from_euler_zyx(-1.57, 0, 0)
                elif object_idx_this_env in [3]:
                    shadow_hand_start_pose.p = gymapi.Vec3(0.38, 0.08, 0.71)
                    object_start_pose.p = gymapi.Vec3(0.0, -0.12, 0.65)
                    object_start_pose.r = gymapi.Quat().from_euler_zyx(0, 1.57, 0)
                elif object_idx_this_env in [4]:
                    shadow_hand_start_pose.p = gymapi.Vec3(0.4, 0.08, 0.72)
                    object_start_pose.p = gymapi.Vec3(0.0, 0, 0.66)
                    object_start_pose.r = gymapi.Quat().from_euler_zyx(1.57, -1.57, 0)

                self.goal_displacement = gymapi.Vec3(-0., 0.0, 5.)
                self.goal_displacement_tensor = to_torch(
                    [self.goal_displacement.x, self.goal_displacement.y, self.goal_displacement.z], device=self.device)
                goal_start_pose = gymapi.Transform()
                goal_start_pose.p = object_start_pose.p + self.goal_displacement

                goal_start_pose.p.z -= 0.0

                table_pose = gymapi.Transform()
                table_pose.p = gymapi.Vec3(0.0, 0.0, 0.5 * self.table_dims.z)
                table_pose.r = gymapi.Quat().from_euler_zyx(-0., 0, 0)

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
                                             scale=self.object_scale[object_idx_this_env])
            self.object_init_state.append([object_start_pose.p.x, object_start_pose.p.y, object_start_pose.p.z,
                                           object_start_pose.r.x, object_start_pose.r.y, object_start_pose.r.z,
                                           object_start_pose.r.w,
                                           0, 0, 0, 0, 0, 0])

            object_dof_props = self.gym.get_actor_dof_properties(env_ptr, object_actor)
            for object_dof_prop in object_dof_props:
                object_dof_prop[4] = 1
                object_dof_prop[5] = 1
                object_dof_prop["stiffness"] = 1
                object_dof_prop["damping"] = 0.5
                object_dof_prop["friction"] = 2.0
            self.gym.set_actor_dof_properties(env_ptr, object_actor, object_dof_props)

            # set friction
            object_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, object_actor)
            for object_shape_prop in object_shape_props:
                object_shape_prop.friction = 0.5
            self.gym.set_actor_rigid_shape_properties(env_ptr, object_actor, object_shape_props)
            self.obj_actors[i].append(object_actor)

            object_body_props = self.gym.get_actor_rigid_body_properties(env_ptr, object_actor)
            for object_body_prop in object_body_props:
                object_body_prop.mass = 0.5
            self.gym.set_actor_rigid_body_properties(env_ptr, object_actor, object_body_props)

            # # add goal object
            # goal_actor = self._load_goal(env_ptr, i, self.goal_asset_dict[object_idx_this_env], goal_start_pose, scale = 1)

            # colorx = random.uniform(0, 1)
            # colory = random.uniform(0, 1)
            # colorz = random.uniform(0, 1)
            # color = gymapi.Vec3(colorx, colory, colorz)
            # # color2 = gymapi.Vec3(0.98, 0.72, 0.6)
            # for o in range(self.num_object_bodies_list[i % len(self.env_dict)]):
            #     self.gym.set_rigid_body_color(env_ptr, object_actor, o, gymapi.MESH_VISUAL, color)

            # add table
            table_handle = self.gym.create_actor(env_ptr, self.table_asset, table_pose, "table", i, -1, 0)
            # self.gym.set_rigid_body_texture(env_ptr, table_handle, 0, gymapi.MESH_VISUAL, table_texture_handle)
            table_idx = self.gym.get_actor_index(env_ptr, table_handle, gymapi.DOMAIN_SIM)
            self.table_indices.append(table_idx)

            # Vision
            self._load_cameras(env_ptr, i, self.camera_props, self.camera_eye_list, self.camera_lookat_list)

            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            self.envs.append(env_ptr)

        self.object_init_state = to_torch(self.object_init_state, device=self.device, dtype=torch.float).view(
            self.num_envs, 13)
        self.goal_states = self.object_init_state.clone()
        self.goal_states[:, self.up_axis_idx] -= 0.04
        self.goal_init_state = self.goal_states.clone()
        self.hand_start_states = to_torch(self.hand_start_states, device=self.device).view(self.num_envs, 13)
        self.fingertip_handles = to_torch(self.fingertip_handles, dtype=torch.long, device=self.device)

        self.hand_indices = to_torch(self.hand_indices, dtype=torch.long, device=self.device)
        self.object_indices = to_torch(self.object_indices, dtype=torch.long, device=self.device)
        self.goal_object_indices = to_torch(self.goal_object_indices, dtype=torch.long, device=self.device)

    def compute_reward(self, actions):
        self.rew_buf[:], self.reset_buf[:], self.reset_goal_buf[:], self.progress_buf[:], self.successes[
                                                                                          :], self.consecutive_successes[
                                                                                              :] = self.compute_hand_reward(
            self.rew_buf, self.reset_buf, self.reset_goal_buf, self.progress_buf, self.successes,
            self.consecutive_successes,
            self.max_episode_length, self.object_dof_pos.squeeze(-1), self.object_dof_vel.squeeze(-1), self.object_pos,
            self.object_rot, self.goal_pos, self.goal_rot, self.right_hand_ff_pos, self.right_hand_mf_pos,
            self.right_hand_rf_pos,
            self.right_hand_lf_pos, self.right_hand_th_pos, self.pen_right_handle_pos, self.pen_left_handle_pos,
            self.dist_reward_scale, self.rot_reward_scale, self.rot_eps, self.actions, self.action_penalty_scale,
            self.success_tolerance, self.reach_goal_bonus, self.fall_dist, self.fall_penalty,
            self.max_consecutive_successes, self.av_factor, (self.object_type == "pen")
        )

        idx = torch.where(self.reset_buf == 1)[0]
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

    def compute_hand_reward(self,
                            rew_buf, reset_buf, reset_goal_buf, progress_buf, successes, consecutive_successes,
                            max_episode_length: float, object_dof_pos, object_dof_vel, object_pos, object_rot,
                            target_pos, target_rot,
                            right_hand_ff_pos, right_hand_mf_pos, right_hand_rf_pos, right_hand_lf_pos,
                            right_hand_th_pos,
                            pen_right_handle_pos, pen_left_handle_pos,
                            dist_reward_scale: float, rot_reward_scale: float, rot_eps: float,
                            actions, action_penalty_scale: float,
                            success_tolerance: float, reach_goal_bonus: float, fall_dist: float,
                            fall_penalty: float, max_consecutive_successes: int, av_factor: float, ignore_z_rot: bool
                            ):

        right_hand_finger_dist = (torch.norm(pen_right_handle_pos - right_hand_ff_pos, p=2, dim=-1) + torch.norm(
            pen_right_handle_pos - right_hand_mf_pos, p=2, dim=-1)
                                  + torch.norm(pen_right_handle_pos - right_hand_rf_pos, p=2, dim=-1) + torch.norm(
                    pen_right_handle_pos - right_hand_lf_pos, p=2, dim=-1)
                                  + torch.norm(pen_right_handle_pos - right_hand_th_pos, p=2, dim=-1))

        right_hand_dist = torch.norm(self.right_hand_pos - pen_right_handle_pos, p=2, dim=-1)

        right_hand_dist_rew = torch.exp(-10 * right_hand_finger_dist) + torch.exp(-10 * right_hand_dist)

        # rot_rew = 1.0/(torch.abs(rot_dist) + rot_eps) * rot_reward_scale

        action_penalty = torch.sum(actions ** 2, dim=-1)

        # Total reward is: position distance + orientation alignment + action regularization + success bonus + fall penalty
        # reward = torch.exp(-0.05*(up_rew * dist_reward_scale)) + torch.exp(-0.05*(right_hand_dist_rew * dist_reward_scale)) + torch.exp(-0.05*(left_hand_dist_rew * dist_reward_scale))
        up_rew = torch.zeros_like(right_hand_dist_rew)
        # up_rew = 10 * object_dof_pos
        up_rew = torch.where(right_hand_finger_dist < 1.5,
                             object_dof_pos + 2 * object_dof_vel,
                             up_rew)

        reward = up_rew + right_hand_dist_rew

        resets = reset_buf
        resets = torch.where(right_hand_dist_rew <= 0, torch.ones_like(reset_buf), reset_buf)
        resets = torch.where(right_hand_finger_dist >= 1.5, torch.ones_like(resets), resets)

        # Find out which envs hit the goal and update successes count
        successes = torch.where(successes == 0,
                                torch.where(object_dof_pos > 0.15,
                                            torch.ones_like(successes), successes), successes)
        # reward = torch.where(successes == 1, reward + 2, reward)

        reward = torch.where(successes > 1, reward + 5, reward)

        resets = torch.where(object_dof_pos > 0.17, torch.ones_like(resets), resets)

        resets = torch.where(progress_buf >= max_episode_length, torch.ones_like(resets), resets)

        goal_resets = torch.zeros_like(resets)

        # num_resets = torch.sum(resets)
        # finished_cons_successes = torch.sum(successes * resets.float())

        # cons_successes = torch.where(num_resets > 0, av_factor*finished_cons_successes/num_resets + (1.0 - av_factor)*consecutive_successes, consecutive_successes)
        cons_successes = torch.where(resets > 0, successes * resets, consecutive_successes)
        return reward, resets, goal_resets, progress_buf, successes, cons_successes

    def compute_observations(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)

        robot_state = self.compute_robot_state(full_obs=False)
        object_state = self.compute_object_state(set_goal=False)
        # base_state = torch.cat((robot_state, object_state), dim=1)
        base_state = robot_state
        base_state = torch.clamp(base_state, -self.cfg["env"]["clip_observations"],
                                 self.cfg["env"]["clip_observations"])

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

        if self.obs_type == 'VisTac':
            # pixel observation
            pixel_obs = self.compute_pixel_obs()
            # force sensor
            touch_force_obs = self.compute_sensor_obs()
            self.obs_states_buf = torch.cat((base_state, pixel_obs, touch_force_obs), dim=1)

        elif self.obs_type == 'TacOnly':
            # force sensor
            touch_force_obs = self.compute_sensor_obs()
            self.obs_states_buf = torch.cat((base_state, touch_force_obs), dim=1)

        elif self.obs_type == 'VisOnly':
            # pixel observation
            pixel_obs = self.compute_pixel_obs()
            self.obs_states_buf = torch.cat((base_state, pixel_obs), dim=1)

        elif self.obs_type == 'Base':
            self.obs_states_buf = base_state

    def compute_sensor_obs(self):
        # forces and torques
        contact = self.contact_force[self.hand_contact_idx].view(self.num_envs, self.num_force_sensors, 3)
        vec_sensor = torch.norm(contact, p=2, dim=2)
        fingertip_force = self.vec_sensor_tensor.view(self.num_envs, 5, 6)[:, :, :3]
        fingertip_force = torch.norm(fingertip_force, p=2, dim=2)
        fingertip_sensor = torch.zeros_like(fingertip_force)
        fingertip_sensor[fingertip_force > 0.3] = 1

        self.sensor_obs = torch.zeros_like(vec_sensor)
        self.sensor_obs[vec_sensor > 0.01] = 1
        self.sensor_obs[:, 0:5] = torch.logical_or(self.sensor_obs[:, 0:5], fingertip_sensor)
        # print(self.sensor_obs[:, :5])
        return self.sensor_obs

    def compute_robot_state(self, full_obs=False):
        # dof_state = self.dof_state.view(self.num_envs, -1, 2)
        # dof_pos = dof_state[..., 0]
        # dof_vel = dof_state[..., 1]
        # robot_state = torch.cat((dof_pos, dof_vel), dim=1)

        robot_qpos = unscale(self.shadow_hand_dof_pos, self.shadow_hand_dof_lower_limits,
                             self.shadow_hand_dof_upper_limits)
        robot_qves = self.shadow_hand_dof_vel#*self.vel_obs_scale
        robot_state = torch.cat((robot_qpos, robot_qves), dim=1)

        self.right_hand_pos = self.rigid_body_states.view(self.num_envs, -1, 13)[:, 3, 0:3]
        self.right_hand_rot = self.rigid_body_states.view(self.num_envs, -1, 13)[:, 3, 3:7]
        self.right_hand_pos = self.right_hand_pos + quat_apply(self.right_hand_rot,
                                                               to_torch([0, 0, 1], device=self.device).repeat(
                                                                   self.num_envs, 1) * 0.08)
        self.right_hand_pos = self.right_hand_pos + quat_apply(self.right_hand_rot,
                                                               to_torch([0, 1, 0], device=self.device).repeat(
                                                                   self.num_envs, 1) * -0.02)

        all_robot_state = robot_state

        if full_obs:
            robot_qpos = unscale(self.shadow_hand_dof_pos, self.shadow_hand_dof_lower_limits,
                                 self.shadow_hand_dof_upper_limits)
            robot_qves = self.vel_obs_scale * self.shadow_hand_dof_vel
            robot_dof_force = self.force_torque_obs_scale * self.dof_force_tensor[:, :24]

            self.fingertip_state = self.rigid_body_states[self.fingertip_indices].view(self.num_envs,
                                                                                       self.num_fingertips, 13)
            self.fingertip_pos = self.fingertip_state[:, :, 0:3]
            num_ft_states = 13 * self.num_fingertips  # 65
            fingertip_state = self.fingertip_state.reshape(self.num_envs, num_ft_states)
            fingertip_force = self.force_torque_obs_scale * self.vec_sensor_tensor[:, :30]
            all_robot_state = torch.cat((robot_qpos, robot_qves, robot_dof_force, fingertip_state, fingertip_force,
                                         self.right_hand_pos,
                                         get_euler_xyz(self.hand_orientations[self.hand_indices, :])[0].unsqueeze(-1),
                                         get_euler_xyz(self.hand_orientations[self.hand_indices, :])[1].unsqueeze(-1),
                                         get_euler_xyz(self.hand_orientations[self.hand_indices, :])[2].unsqueeze(-1),
                                         self.actions), dim=1)

        return all_robot_state

    def compute_object_state(self, set_goal=True):
        self.object_pose = self.root_state_tensor[self.object_indices, 0:7]
        self.object_pos = self.root_state_tensor[self.object_indices, 0:3]
        self.object_rot = self.root_state_tensor[self.object_indices, 3:7]
        self.object_linvel = self.root_state_tensor[self.object_indices, 7:10]
        self.object_angvel = self.root_state_tensor[self.object_indices, 10:13]

        self.goal_pose = self.goal_states[:, 0:7]
        self.goal_pos = self.goal_states[:, 0:3]
        self.goal_rot = self.goal_states[:, 3:7]

        self.slide_pos = self.rigid_body_states.view(self.num_envs, -1, 13)[:, 46 + 1, 0:3]

        # self.pen_right_handle_pos = self.rigid_body_states.view(self.num_envs, -1, 13)[:, 46 + 1, 0:3]
        # self.pen_right_handle_rot = self.rigid_body_states.view(self.num_envs, -1, 13)[:, 46 + 1, 3:7]
        # self.pen_right_handle_pos[:, 1] += 0.08
        # self.pen_right_handle_pos = self.pen_right_handle_pos + quat_apply(self.pen_right_handle_rot,
        #                                                                    to_torch([0, 1, 0],
        #                                                                             device=self.device).repeat(
        #                                                                        self.num_envs, 1) * -0.1)
        # self.pen_right_handle_pos = self.pen_right_handle_pos + quat_apply(self.pen_right_handle_rot,
        #                                                                    to_torch([1, 0, 0],
        #                                                                             device=self.device).repeat(
        #                                                                        self.num_envs, 1) * 0.0)
        # self.pen_right_handle_pos = self.pen_right_handle_pos + quat_apply(self.pen_right_handle_rot,
        #                                                                    to_torch([0, 0, 1],
        #                                                                             device=self.device).repeat(
        #                                                                        self.num_envs, 1) * 0.0)

        self.pen_left_handle_pos = self.rigid_body_states.view(self.num_envs, -1, 13)[:, 46, 0:3]
        self.pen_left_handle_rot = self.rigid_body_states.view(self.num_envs, -1, 13)[:, 46, 3:7]
        self.pen_left_handle_pos = self.pen_left_handle_pos + quat_apply(self.pen_left_handle_rot,
                                                                         to_torch([0, 1, 0], device=self.device).repeat(
                                                                             self.num_envs, 1) * 0.07)
        self.pen_left_handle_pos = self.pen_left_handle_pos + quat_apply(self.pen_left_handle_rot,
                                                                         to_torch([1, 0, 0], device=self.device).repeat(
                                                                             self.num_envs, 1) * 0.0)
        self.pen_left_handle_pos = self.pen_left_handle_pos + quat_apply(self.pen_left_handle_rot,
                                                                         to_torch([0, 0, 1], device=self.device).repeat(
                                                                             self.num_envs, 1) * 0.0)

        object_state = torch.cat((self.object_pose, self.object_linvel, self.vel_obs_scale * self.object_angvel,
                                  self.pen_left_handle_pos, self.pen_right_handle_pos,
                                  ), dim=1)

        if set_goal:
            goal_state = torch.cat((self.goal_pose, quat_mul(self.object_rot, quat_conjugate(self.goal_rot)),), dim=1)
            all_object_state = torch.cat((object_state, goal_state), dim=1)
        else:
            all_object_state = object_state
        return all_object_state

    def reset_idx(self, env_ids, goal_env_ids):
        """
        Reset and randomize the environment

        Args:
            env_ids (tensor): The index of the environment that needs to reset

            goal_env_ids (tensor): The index of the environment that only goals need reset

        """
        # randomization can happen only at reset time, since it can reset actor positions on GPU
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

        # generate random values
        rand_floats = torch_rand_float(-1.0, 1.0, (len(env_ids), self.num_shadow_hand_dofs * 2 + 3), device=self.device)

        # reset object
        self.root_state_tensor[self.object_indices[env_ids]] = self.object_init_state[env_ids].clone()
        self.root_state_tensor[self.object_indices[env_ids], 0:2] = self.object_init_state[env_ids, 0:2] + \
                                                                    self.reset_position_noise * rand_floats[:, 0:2]
        self.root_state_tensor[self.object_indices[env_ids], self.up_axis_idx] = self.object_init_state[
                                                                                     env_ids, self.up_axis_idx] + \
                                                                                 self.reset_position_noise * rand_floats[
                                                                                                             :,
                                                                                                             self.up_axis_idx]

        new_object_rot = randomize_rotation(rand_floats[:, 3], rand_floats[:, 4], self.x_unit_tensor[env_ids],
                                            self.y_unit_tensor[env_ids])
        if self.object_type == "pen":
            rand_angle_y = torch.tensor(0.3)
            new_object_rot = randomize_rotation_pen(rand_floats[:, 3], rand_floats[:, 4], rand_angle_y,
                                                    self.x_unit_tensor[env_ids], self.y_unit_tensor[env_ids],
                                                    self.z_unit_tensor[env_ids])

        # self.root_state_tensor[self.object_indices[env_ids], 3:7] = new_object_rot
        self.root_state_tensor[self.object_indices[env_ids], 7:13] = torch.zeros_like(
            self.root_state_tensor[self.object_indices[env_ids], 7:13])

        object_indices = torch.unique(torch.cat([self.object_indices[env_ids],
                                                 ]).to(torch.int32))
        # self.gym.set_actor_root_state_tensor_indexed(self.sim,
        #                                              gymtorch.unwrap_tensor(self.root_state_tensor),
        #                                              gymtorch.unwrap_tensor(object_indices), len(object_indices))


        # reset shadow hand
        hand_indices = self.hand_indices[env_ids].to(torch.int32)
        self._reset_hand(env_ids, rand_floats, self.hand_indices[env_ids])

        self.object_dof_pos[env_ids, :] = to_torch([0], device=self.device)
        self.object_dof_vel[env_ids, :] = to_torch([0], device=self.device)

        self.prev_targets[env_ids, self.num_shadow_hand_dofs:self.num_shadow_hand_dofs + 1] = to_torch([0],
                                                                                                       device=self.device)
        self.cur_targets[env_ids, self.num_shadow_hand_dofs:self.num_shadow_hand_dofs + 1] = to_torch([0],
                                                                                                      device=self.device)
        all_hand_indices = torch.unique(torch.cat([hand_indices,
                                                   ]).to(torch.int32))

        all_indices = torch.unique(torch.cat([all_hand_indices,
                                              object_indices]).to(torch.int32))

        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(all_indices), len(all_indices))

        self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self.prev_targets),
                                                        gymtorch.unwrap_tensor(all_indices), len(all_indices))

        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_state_tensor),
                                                     gymtorch.unwrap_tensor(all_indices), len(all_indices))
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
            self.cur_targets[:, self.actuated_dof_indices] = scale(self.actions[:,],
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

            self.apply_forces[:, 1, :] = self.actions[:, 0:3] * self.dt * self.transition_scale * 100000
            self.apply_torque[:, 1, :] = self.actions[:, 3:6] * self.dt * self.orientation_scale * 1000
            self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(self.apply_forces),
                                                    gymtorch.unwrap_tensor(self.apply_torque), gymapi.ENV_SPACE)

        self.prev_targets[:, self.actuated_dof_indices] = self.cur_targets[:, self.actuated_dof_indices]

        # self.prev_targets[:, 49] = self.cur_targets[:, 49]
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.cur_targets))

    def _reset_hand(self, env_ids, rand_floats, hand_indices):

        delta_max = self.shadow_hand_dof_upper_limits - self.shadow_hand_dof_default_pos
        delta_min = self.shadow_hand_dof_lower_limits - self.shadow_hand_dof_default_pos
        rand_delta = delta_min + (delta_max - delta_min) * 0.5 * (rand_floats[:, :self.num_shadow_hand_dofs] + 1)

        dof_pos = self.shadow_hand_default_dof_pos  +  self.reset_dof_pos_noise * rand_delta
        self.shadow_hand_dof_pos[env_ids, :] = dof_pos
        self.shadow_hand_dof_vel[env_ids, :] = self.shadow_hand_dof_default_vel  + self.reset_dof_vel_noise * rand_floats[:, self.num_shadow_hand_dofs:self.num_shadow_hand_dofs * 2]
        self.prev_targets[env_ids, :self.num_shadow_hand_dofs] = dof_pos
        self.cur_targets[env_ids, :self.num_shadow_hand_dofs] = dof_pos

        self.hand_positions[hand_indices, :] = self.saved_root_tensor[hand_indices, 0:3] + self.startPositionNoise * rand_floats[:, -3:]
        self.hand_orientations[hand_indices, :] = self.saved_root_tensor[hand_indices, 3:7]
        self.hand_linvels[hand_indices, :] = 0
        self.hand_angvels[hand_indices, :] = 0

        return


#####################################################################
###=========================jit functions=========================###
#####################################################################


# @torch.jit.script


@torch.jit.script
def randomize_rotation(rand0, rand1, x_unit_tensor, y_unit_tensor):
    return quat_mul(quat_from_angle_axis(rand0 * np.pi, x_unit_tensor),
                    quat_from_angle_axis(rand1 * np.pi, y_unit_tensor))


@torch.jit.script
def randomize_rotation_pen(rand0, rand1, max_angle, x_unit_tensor, y_unit_tensor, z_unit_tensor):
    rot = quat_mul(quat_from_angle_axis(0.5 * np.pi + rand0 * max_angle, x_unit_tensor),
                   quat_from_angle_axis(rand0 * np.pi, z_unit_tensor))
    return rot