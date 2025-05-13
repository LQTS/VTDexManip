import numpy as np
import os
import torch
from tv_tasks.utils.torch_jit_utils import *
from tv_tasks.tasks.base.shadow_hand import ShadowHandBase
from isaacgym import gymtorch
from isaacgym import gymapi
import random
import copy

class ScrewFaucet(ShadowHandBase):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        self.table_dims = gymapi.Vec3(1, 1, 0.6)
        self.set_camera(cfg, self.table_dims.z)

        self.transition_scale = cfg["env"]["transition_scale"]
        self.orientation_scale = cfg["env"]["orientation_scale"]
        self.reward_weight = cfg["env"]["reward_weight"]
        print(f"reward_weight --> {self.reward_weight}")
        self.enable_touch_reward = cfg["env"]["enableTouchReward"]
        print(f"enable_touch_reward --> {self.enable_touch_reward}")
        self.enable_distance_reward = cfg["env"]["enableDisReward"]
        print(f"enable_distance_reward --> {self.enable_distance_reward}")
        self.tactile_theshold = cfg["env"].get("tac_theshold", 0.01)
        print(f"tactile_theshold --> {self.tactile_theshold}")
        self.masked_modality = cfg["env"].get("masked_modality", None)

        super().__init__(cfg, sim_params, physics_engine, device_type, device_id, headless)
        # # self.rigid_body_states= self.rigid_body_states.view(self.num_envs, -1, 13)
        # self.num_bodies = self.rigid_body_states.view(self.num_envs, -1, 13).shape[1]
        # self.apply_forces = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device, dtype=torch.float)
        # self.apply_torque = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device, dtype=torch.float)
        # self.placeholder = torch.zeros([3, 1024, 1024, 1024], dtype=torch.float).cuda(device_id)  #3*4GB GPU MEMORY

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

        self.shadow_hand_asset = self.gym.load_asset(self.sim, self.asset_root, shadow_hand_asset_file, self.hand_asset_options)

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

        object_init_height = self.cfg["env"]["object_init_height"]
        hand_init_height = self.cfg["env"]["hand_init_height"]
        object_scale = self.cfg["env"]["object_scale"]

        self.object_idx = []
        self.num_object_bodies_list = []
        self.num_object_shapes_list = []

        self.object_init_height_dict = {}
        self.hand_init_height_dict = {}
        self.object_scale = {}
        self.object_asset_dict = {}
        self.goal_asset_dict = {}
        self.obj_maniputed_idx = []

        for object_id, object_code in enumerate(self.env_dict):
            # load manipulated object and goal assets
            object_asset_options = gymapi.AssetOptions()
            object_asset_options.density = 500
            object_asset_options.fix_base_link = True
            # object_asset_options.collapse_fixed_joints = True
            object_asset_options.disable_gravity = True
            object_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE

            object_code = str(object_code)
            object_asset_file = "mobility.urdf"
            object_asset = self.gym.load_asset(self.sim, self.obj_asset_root + object_code,
                                                      object_asset_file, object_asset_options)

            object_asset_options.disable_gravity = True
            goal_asset = self.gym.load_asset(self.sim, self.obj_asset_root + object_code,
                                             object_asset_file, object_asset_options)

            self.object_asset_dict[object_id] = object_asset
            self.goal_asset_dict[object_id] = goal_asset
            self.object_init_height_dict[object_id] = object_init_height[object_id]
            self.hand_init_height_dict[object_id] = hand_init_height[object_id]
            self.object_scale[object_id] = object_scale[object_id]
            self.object_idx.append(object_id)

            self.num_object_bodies_list.append(self.gym.get_asset_rigid_body_count(object_asset))
            self.num_object_shapes_list.append(self.gym.get_asset_rigid_shape_count(object_asset))
            self.obj_maniputed_idx.append(self.gym.find_asset_rigid_body_index(object_asset, "object"))
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
        self.obj_contact_idx_one_env = []
        self.arm_indices = []

        self.hand_contact_idx_one_env = []

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

            shadow_hand_start_pose = gymapi.Transform()
            shadow_hand_start_pose.p = gymapi.Vec3(0.38, 0.11, 0.715)
            shadow_hand_start_pose.r = gymapi.Quat().from_euler_zyx(3.14159, 0, 1.57)

            object_start_pose = gymapi.Transform()
            self.object_rise = self.object_init_height_dict[object_idx_this_env]
            object_start_pose.p = gymapi.Vec3(0.0, 0.0,
                                              self.table_dims.z + self.object_rise)  # gymapi.Vec3(0.0, 0.0, 0.72)
            object_start_pose.r = gymapi.Quat().from_euler_zyx(0, 0, 3.14)

            if object_idx_this_env % 5 in [0]:
                shadow_hand_start_pose.p = gymapi.Vec3(0.32, 0.12, 0.72)
            elif object_idx_this_env % 5 in [1]:
                shadow_hand_start_pose.p = gymapi.Vec3(0.38, 0.12, 0.74)
            elif object_idx_this_env % 5 in [2]:
                shadow_hand_start_pose.p = gymapi.Vec3(0.36, -0.01, 0.75)
            elif object_idx_this_env % 5 in [3]:
                shadow_hand_start_pose.p = gymapi.Vec3(0.37, -0.015, 0.73)
                object_start_pose.r = gymapi.Quat().from_euler_zyx(0, 1.57, 0)
            elif object_idx_this_env % 5 in [4]:
                shadow_hand_start_pose.p = gymapi.Vec3(0.37, 0.15, 0.71)
                object_start_pose.r = gymapi.Quat().from_euler_zyx(0, 0, 1.57)
            # elif object_idx_this_env in [5]:
            #     shadow_hand_start_pose.p = gymapi.Vec3(0.375, 0.08, 0.74)
            # elif object_idx_this_env in [6]:
            #     shadow_hand_start_pose.p = gymapi.Vec3(0.4, 0.012, 0.89)
            # elif object_idx_this_env in [4]:
            #     shadow_hand_start_pose.p = gymapi.Vec3(0.4, 0.012, 0.82)

            # if object_idx_this_env in [0]:
            #     shadow_hand_start_pose.p = gymapi.Vec3(0.37, -0.015, 0.72)
            # elif object_idx_this_env in [1]:
            #     shadow_hand_start_pose.p = gymapi.Vec3(0.37, -0.015, 0.74)
            # elif object_idx_this_env in [2]:
            #     shadow_hand_start_pose.p = gymapi.Vec3(0.37, -0.015, 0.75)
            # elif object_idx_this_env in [3]:
            #     shadow_hand_start_pose.p = gymapi.Vec3(0.37, -0.015, 0.73)
            #     object_start_pose.r = gymapi.Quat().from_euler_zyx(0, 1.57, 0)
            # elif object_idx_this_env in [4]:
            #     shadow_hand_start_pose.p = gymapi.Vec3(0.37, -0.015, 0.71)
            #     object_start_pose.r = gymapi.Quat().from_euler_zyx(0, 0, 1.57)

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
            object_actor = self._load_object(env_ptr, i, self.object_asset_dict[object_idx_this_env], object_start_pose, scale = self.object_scale[object_idx_this_env])
            self.object_init_state.append([object_start_pose.p.x, object_start_pose.p.y, object_start_pose.p.z,
                                           object_start_pose.r.x, object_start_pose.r.y, object_start_pose.r.z,
                                           object_start_pose.r.w,
                                           0, 0, 0, 0, 0, 0])

            object_dof_props = self.gym.get_actor_dof_properties(env_ptr, object_actor)
            for object_dof_prop in object_dof_props:
                # object_dof_prop[4] = 5
                # object_dof_prop[5] = 5
                object_dof_prop["stiffness"] = 0
                object_dof_prop["damping"] = 0.1
                object_dof_prop["friction"] = 0.5
            self.gym.set_actor_dof_properties(env_ptr, object_actor, object_dof_props)

            object_body_props = self.gym.get_actor_rigid_body_properties(env_ptr, object_actor)
            for object_body_prop in object_body_props:
                object_body_prop.mass = 0.15
            self.gym.set_actor_rigid_body_properties(env_ptr, object_actor, object_body_props)

            # # set friction
            # object_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, object_actor)
            # for object_shape_prop in object_shape_props:
            #     object_shape_prop.friction = 1
            # self.gym.set_actor_rigid_shape_properties(env_ptr, object_actor, object_shape_props)
            self.obj_actors[i].append(object_actor)

            # # add goal object
            # goal_actor = self._load_goal(env_ptr, i, self.goal_asset_dict[object_idx_this_env], goal_start_pose, scale = 1)

            # # set color
            # for o in range(self.num_object_bodies_list[i % len(self.env_dict)]):
            #     self.gym.set_rigid_body_color(env_ptr, object_actor, o, gymapi.MESH_VISUAL, gymapi.Vec3(0.6, 0.72, 0.98))
            #     self.gym.set_rigid_body_color(env_ptr, goal_actor, o, gymapi.MESH_VISUAL, gymapi.Vec3(0.6, 0.72, 0.98))

            # add table
            table_handle = self.gym.create_actor(env_ptr, self.table_asset, table_pose, "table", i, -1, 0)
            # self.gym.set_rigid_body_texture(env_ptr, table_handle, 0, gymapi.MESH_VISUAL, table_texture_handle)

            # color = gymapi.Vec3(0.9, 0.6, 0.3)
            # # # # set color
            # # # colorx0 = random.uniform(0, 1)
            # # # colory0 = random.uniform(0, 1)
            # # # colorz0 = random.uniform(0, 1)
            # # # color0 = gymapi.Vec3(colorx0, colory0, colorz0)
            # self.gym.set_rigid_body_color(env_ptr, table_handle, 0, gymapi.MESH_VISUAL, color)

            table_idx = self.gym.get_actor_index(env_ptr, table_handle, gymapi.DOMAIN_SIM)
            self.table_indices.append(table_idx)

            # Vision
            self._load_cameras(env_ptr, i, self.camera_props, self.camera_eye_list, self.camera_lookat_list)

            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            self.obj_contact_idx_one_env.append(self.gym.find_actor_rigid_body_index(env_ptr, object_actor, "object", gymapi.DOMAIN_ENV))
            self.envs.append(env_ptr)

        self.hand_contact_idx_one_env = self.hand_contact_idx[:20]
        self.object_init_state = to_torch(self.object_init_state, device=self.device, dtype=torch.float).view(self.num_envs, 13)
        self.goal_states = self.object_init_state.clone()
        self.goal_states[:, self.up_axis_idx] -= 0.04
        self.goal_init_state = self.goal_states.clone()
        self.hand_start_states = to_torch(self.hand_start_states, device=self.device).view(self.num_envs, 13)
        self.fingertip_handles = to_torch(self.fingertip_handles, dtype=torch.long, device=self.device)

        self.hand_indices = to_torch(self.hand_indices, dtype=torch.long, device=self.device)
        self.object_indices = to_torch(self.object_indices, dtype=torch.long, device=self.device)
        self.goal_object_indices = to_torch(self.goal_object_indices, dtype=torch.long, device=self.device)



    def compute_reward(self, actions):
        self.rew_buf[:], self.reset_buf[:], self.reset_goal_buf[:], self.progress_buf[:], self.successes[:], self.consecutive_successes[:] = self.compute_hand_reward(
            self.rew_buf, self.reset_buf, self.reset_goal_buf, self.progress_buf, self.successes, self.consecutive_successes,
            self.max_episode_length, self.object_dof_pos.squeeze(-1), self.object_dof_vel.squeeze(-1), self.object_pos, self.object_rot, self.goal_pos, self.goal_rot, self.right_hand_ff_pos, self.right_hand_mf_pos, self.right_hand_rf_pos,
            self.right_hand_lf_pos, self.right_hand_th_pos,
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
            print("Direct average consecutive successes = {:.1f}".format(direct_average_successes/(self.total_resets + self.num_envs)))
            if self.total_resets > 0:
                print("Post-Reset average consecutive successes = {:.1f}".format(self.total_successes/self.total_resets))

    def compute_hand_reward(self,
                             rew_buf, reset_buf, reset_goal_buf, progress_buf, successes, consecutive_successes,
                             max_episode_length: float,object_dof_pos, object_dof_vel, object_pos, object_rot, target_pos, target_rot,
                                                        right_hand_ff_pos, right_hand_mf_pos, right_hand_rf_pos, right_hand_lf_pos, right_hand_th_pos,
                                                        dist_reward_scale: float, rot_reward_scale: float, rot_eps: float,
    actions, action_penalty_scale: float,
    success_tolerance: float, reach_goal_bonus: float, fall_dist: float,
    fall_penalty: float, max_consecutive_successes: int, av_factor: float, ignore_z_rot: bool
    ):
        a, b, c, d = self.reward_weight

        r1 = torch.min(object_dof_pos, torch.tensor(7.0))
        r2 = torch.clamp(object_dof_vel, -10.0, 10.0)

        action_penalty = torch.sum(actions ** 2, dim=-1)

        if self.enable_touch_reward:
            contacts = self.match_contacts2().to(torch.float)
        else:
            contacts = torch.ones(self.sensor_obs.shape[0]).to(self.device)
        r2 = torch.where(r2 > 0, torch.where(contacts > 0, r2, torch.zeros_like(r2)), r2)
        # reward = 0.5 * r1 + r2
        reward = a * r1 + b * r2  + d*action_penalty
        # print(right_hand_finger_dist)
        # print('hhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh')
        # print(right_hand_dist_rew)
        # reward = r1 + 0.5 * right_hand_dist_rew
        # reward = torch.where(reward > 3.14 * 2, reward + 10, torch.where(reward > 3.14, reward + 5, reward))
        #reward = torch.where(r1 > 3, reward + 10, torch.where(r1 > 2, reward + 5, reward))
        if self.enable_distance_reward:
            fingertips = torch.stack([right_hand_ff_pos, right_hand_mf_pos, right_hand_rf_pos, right_hand_lf_pos, right_hand_th_pos], dim=1)
            fingertips_z = fingertips[..., -1]  # table height 0.6
            target_dim_z = self.faucet_pos[..., -1]  # a - 0.05 == the top of obj
            target_dim_z = target_dim_z.repeat(fingertips_z.shape[0] // target_dim_z.shape[0]).view(-1, 1)
            dis_z = torch.norm(target_dim_z - fingertips_z, p=1, dim=1)
            r_d = torch.exp(-10 * dis_z)  # 0~1
            # fingertips[..., -1] -= 0.6
            # target_dim_z = to_torch(list(self.hand_init_height_dict.values()),device=self.device) - 0.05 - 0.02  # a - 0.05 == the top of obj
            # target = torch.cat([torch.zeros((target_dim_z.shape[0], 2), dtype=torch.float, device=self.device),
            #                     target_dim_z.view(-1, 1)], dim=1)
            # target = target.repeat(fingertips.shape[0]//target.shape[0], 1).view(-1, 1, 3)
            # dis = torch.norm(fingertips-target, p=2, dim=-1).sum(dim=1)
            # r_d = torch.exp(-20 * dis)  # 0~1
            # r_d = -torch.log(4 * dis_z + 1)
            reward += c * r_d

        resets = reset_buf

        # # Find out which envs hit the goal and update successes count
        resets = torch.where(r1 > 6.15, torch.ones_like(resets), resets)
        resets = torch.where(progress_buf >= max_episode_length, torch.ones_like(resets), resets)
        # tanh 2
        # reward = torch.where((resets == 1) & (progress_buf < max_episode_length), reward + 100, reward)

        goal_resets = resets
        # if any(successes):
        #     print('aa')
        successes = torch.where((r1 > 6), torch.ones_like(successes), successes)
        # if any(successes):
        #     print('aa')
        # success bonus tanh3
        reward = torch.where(successes > 0, reward + 5, reward)

        # reward = torch.where((resets == 1) & (progress_buf < max_episode_length), reward + 10, reward)

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
        base_state = torch.clamp(base_state, -self.cfg["env"]["clip_observations"], self.cfg["env"]["clip_observations"])

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
        touch_force_obs = self.compute_sensor_obs()
        if self.obs_type == 'VisTac':
            if self.masked_modality == "vis":
                pixel_obs = torch.flatten(torch.zeros_like(self.img_buf), start_dim=1, end_dim=-1)
            elif self.masked_modality == "tac":
                touch_force_obs = torch.zeros_like(touch_force_obs)
                # pixel observation
                pixel_obs = self.compute_pixel_obs()
            else:
                # pixel observation
                pixel_obs = self.compute_pixel_obs()
            # force sensor
            # touch_force_obs = self.compute_sensor_obs()
            self.obs_states_buf = torch.cat((base_state, pixel_obs, touch_force_obs), dim=1)
            # pixel observation
            # pixel_obs = self.compute_pixel_obs()
            # # force sensor
            # # touch_force_obs = self.compute_sensor_obs()
            # self.obs_states_buf = torch.cat((base_state, pixel_obs, touch_force_obs), dim=1)

        elif self.obs_type == 'TacOnly':
            # force sensor
            # touch_force_obs = self.compute_sensor_obs()
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
        # vec_sensor = self.vec_sensor_tensor
        vec_sensor = contact
        vec_sensor = torch.norm(vec_sensor, p=2, dim=2)
        self.sensor_obs = torch.zeros_like(vec_sensor)
        self.sensor_obs[vec_sensor > self.tactile_theshold] = 1
        # print(vec_sensor)
        return self.sensor_obs

    def compute_robot_state(self, full_obs=False):
        # dof_state = self.dof_state.view(self.num_envs, -1, 2)
        # dof_pos = dof_state[..., 0]
        # dof_vel = dof_state[..., 1]
        # robot_state = torch.cat((dof_pos, dof_vel), dim=1)

        robot_qpos = unscale(self.shadow_hand_dof_pos, self.shadow_hand_dof_lower_limits,
                             self.shadow_hand_dof_upper_limits)
        robot_qvel = self.vel_obs_scale * self.shadow_hand_dof_vel
        robot_state = torch.cat((robot_qpos, robot_qvel), dim=1)

        self.right_hand_pos = self.rigid_body_states[self.palm_body_idx][:, 0:3]
        self.right_hand_rot = self.rigid_body_states[self.palm_body_idx][:, 3:7]
        self.right_hand_pos = self.right_hand_pos + quat_apply(self.right_hand_rot,
                                                               to_torch([0, 0, 1], device=self.device).repeat(
                                                                   self.num_envs, 1) * 0.13)
        self.right_hand_pos = self.right_hand_pos + quat_apply(self.right_hand_rot,
                                                               to_torch([0, 1, 0], device=self.device).repeat(
                                                                   self.num_envs, 1) * -0.04)

        all_robot_state = robot_state

        if full_obs:
            robot_qpos = unscale(self.shadow_hand_dof_pos, self.shadow_hand_dof_lower_limits,
                                 self.shadow_hand_dof_upper_limits)
            robot_qves = self.vel_obs_scale * self.shadow_hand_dof_vel
            robot_dof_force = self.force_torque_obs_scale * self.dof_force_tensor[:, :24]

            self.fingertip_state = self.rigid_body_states[self.fingertip_indices].view(self.num_envs, self.num_fingertips, 13)
            self.fingertip_pos = self.fingertip_state[:,:, 0:3]
            num_ft_states = 13 * self.num_fingertips  # 65
            fingertip_state = self.fingertip_state.reshape(self.num_envs, num_ft_states)
            fingertip_force = self.force_torque_obs_scale * self.vec_sensor_tensor[:, :30]
            all_robot_state = torch.cat((robot_qpos, robot_qves, robot_dof_force, fingertip_state, fingertip_force,
                                         get_euler_xyz(self.hand_orientations[self.hand_indices, :])[0].unsqueeze(-1),
                                         get_euler_xyz(self.hand_orientations[self.hand_indices, :])[1].unsqueeze(-1),
                                         get_euler_xyz(self.hand_orientations[self.hand_indices, :])[2].unsqueeze(-1),
                                         self.actions), dim=1)

        return all_robot_state

    # def compute_robot_state(self, full_obs=False):
    #     # dof_state = self.dof_state.view(self.num_envs, -1, 2)
    #     robot_qpos = unscale(self.shadow_hand_dof_pos, self.shadow_hand_dof_lower_limits,
    #                          self.shadow_hand_dof_upper_limits)
    #     robot_qvel = self.vel_obs_scale * self.shadow_hand_dof_vel
    #
    #     obj_qpos = self.object_dof_pos
    #     obj_qvel = self.vel_obs_scale * self.object_dof_vel
    #     all_robot_state = torch.cat((robot_qpos, robot_qvel, obj_qpos, obj_qvel), dim=1)
    #     return all_robot_state

    def compute_object_state(self, set_goal=True):
        self.object_pose = self.root_state_tensor[self.object_indices, 0:7]
        self.object_pos = self.root_state_tensor[self.object_indices, 0:3]
        self.object_rot = self.root_state_tensor[self.object_indices, 3:7]
        self.object_linvel = self.root_state_tensor[self.object_indices, 7:10]
        self.object_angvel = self.root_state_tensor[self.object_indices, 10:13]

        self.goal_pose = self.goal_states[:, 0:7]
        self.goal_pos = self.goal_states[:, 0:3]
        self.goal_rot = self.goal_states[:, 3:7]

        object_state = torch.cat((self.object_pose, self.object_linvel, self.vel_obs_scale * self.object_angvel,
                                  ), dim=1)

        assert len(self.env_dict) % 5 == 0, "if length is not 5, please change line 604"
        self.faucet_pos = copy.deepcopy(self.rigid_body_states[self.obj_contact_idx][:, 0:3])
        self.faucet_pos[:, 2] -= torch.tensor([0.02, 0.01, -0.08, 0.02, 0.01,], dtype=torch.float, device=self.device).repeat(self.faucet_pos.shape[0]//5)

        if set_goal:
            goal_state = torch.cat((self.goal_pose, quat_mul(self.object_rot, quat_conjugate(self.goal_rot)),), dim=1)
            all_object_state = torch.cat((object_state, goal_state), dim=1)
        else:
            all_object_state = object_state
        return  all_object_state

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
            targets = self.prev_targets[:, self.actuated_dof_indices] + self.shadow_hand_dof_speed_scale * self.dt * self.actions
            self.cur_targets[:, self.actuated_dof_indices] = tensor_clamp(targets,
                                                                          self.shadow_hand_dof_lower_limits[self.actuated_dof_indices], self.shadow_hand_dof_upper_limits[self.actuated_dof_indices])
        else:
            self.cur_targets[:, self.actuated_dof_indices] = scale(self.actions[:, :],
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


            # self.apply_forces[:, 1, :] = self.actions[:, 0:3] * self.dt * self.transition_scale * 100000
            # self.apply_torque[:, 1, :] = self.actions[:, 3:6] * self.dt * self.orientation_scale * 1000
            # self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(self.apply_forces),
            #                                         gymtorch.unwrap_tensor(self.apply_torque), gymapi.ENV_SPACE)

        self.prev_targets[:, self.actuated_dof_indices] = self.cur_targets[:, self.actuated_dof_indices]

        # self.prev_targets[:, 49] = self.cur_targets[:, 49]
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.cur_targets))

    def match_contacts2(self, min_contacts=1):

        all_obj_contact_idx_one_env = torch.tensor(self.obj_contact_idx_one_env).repeat(self.num_envs//self.obj_contact_idx_one_env.__len__())#.view(-1, self.obj_contact_idx_one_env.__len__())

        num_contacts_in_envs = self.sensor_obs.sum(dim=1)
        contact_envs = num_contacts_in_envs >= min_contacts
        contact_in_envs = torch.zeros_like(self.sensor_obs).to(torch.bool)
        for i, contact_happen_env in enumerate(contact_envs):
            contact_obj_idx = all_obj_contact_idx_one_env[i]
            contact_pairs, _ = torch.cat([torch.tensor(self.hand_contact_idx_one_env).view(-1, 1),
                                          contact_obj_idx.repeat(
                                              self.hand_contact_idx_one_env.__len__()).view(-1, 1)], dim=-1).sort(dim=1,
                                                                                                                  descending=True)

            if contact_happen_env:
                gym_contact = self.gym.get_env_rigid_contacts(self.envs[i])
                all_contact_pairs = to_torch(np.vstack([gym_contact["body0"], gym_contact["body1"]]), dtype=torch.int32, device=self.device).T
                contact_idxs = torch.where(self.sensor_obs[i] > 0) # get hand contact idx

                set1 = set(map(tuple, contact_pairs[contact_idxs].tolist()))
                set2 = set(map(tuple, all_contact_pairs.tolist()))
                # Find the intersection
                intersection = set1.intersection(set2)
                # Get the indices of the intersection in tensor1
                indices_in_tensor1 = [contact_pairs.tolist().index(list(row)) for row in intersection]
                contact_in_envs[i, indices_in_tensor1] = True
        return contact_in_envs.sum(dim=1)>=min_contacts

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