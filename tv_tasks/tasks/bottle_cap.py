# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
import torch

from tv_tasks.utils.torch_jit_utils import *
from tv_tasks.tasks.base.shadow_hand import ShadowHandBase
from isaacgym import gymtorch
from isaacgym import gymapi
import pickle
from isaacgym.gymutil import get_property_setter_map, get_property_getter_map, get_default_setter_args, apply_random_samples, check_buckets, generate_random_samples
import operator

class BottleCap(ShadowHandBase):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        self.table_dims = gymapi.Vec3(1, 1, 0.6)
        self.set_camera(cfg, self.table_dims.z)
        self.base_obs_dim = list(cfg["env"]["obs_dim"].values())[0]
        print(f"RandomizeMode --> {cfg['task']['randomize']}")
        self.enable_distance_reward = cfg["env"]["enableDisReward"]
        print(f"enable_distance_reward --> {self.enable_distance_reward}")
        self.enable_touch_reward = cfg["env"]["enableTouchReward"]
        print(f"enable_touch_reward --> {self.enable_touch_reward}")
        self.enable_soft_contact = cfg["env"]["enable_soft_contact"]
        print(f"enable_soft_contact --> {self.enable_soft_contact}")
        self.min_contacts = cfg["env"]["min_contacts"]
        print(f"min_contacts --> {self.min_contacts}")
        self.reward_weight = cfg["env"]["reward_weight"]
        print(f"reward_weight --> {self.reward_weight}")
        self.hand_bias = cfg["env"]["hand_bias"]
        print(f"hand_bias --> {self.hand_bias}")
        self.success_type = cfg["env"]["success_type"]
        print(f"success_type --> {self.success_type}")
        self.tactile_theshold = cfg["env"].get("tac_theshold", 0.01)
        print(f"tactile_theshold --> {self.tactile_theshold}")
        self.masked_modality = cfg["env"].get("masked_modality", None)
        self.tac_noise = cfg["env"].get("tac_noise", None)
        print(f"tac_noise --> {self.tac_noise}")
        self.hysteresis = cfg["env"].get("hysteresis", False)
        print(f"hysteresis --> {self.hysteresis}")
        self.last_qpos = 0.0
        self.last_qpos_init = None
        self.action_penalty = []
        self.action_penalty1 = []

        super().__init__(cfg, sim_params, physics_engine, device_type, device_id, headless)
        # create some wrapper tensors for different slices
        # self.reset(torch.arange(self.num_envs, device=self.device))
        self.last_actions = torch.zeros((self.num_envs, self.num_actions), dtype=torch.float, device=self.device)
        self.soft_contact = torch.zeros((self.num_envs, 10), dtype=torch.float, device=self.device)

        # self.placeholder = torch.zeros([3, 1024, 1024, 1024], dtype=torch.float).cuda(device_id)  # 3*4GB GPU MEMORY
    def _create_envs(self, num_envs, spacing, num_per_row):

        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        # load assets
        self._create_hand_asset()
        self._create_obj_asset()
        self._create_table_asset(self.table_dims)

        self.bottle_cap_idx = 47
        self.hand_contact_idx_one_env = []
        self.envs_config()

        self.object_idx = to_torch(self.object_idx, dtype=torch.int32, device=self.device)
        object_idx_list = [idx.item() for idx in self.object_idx]
        self.obj_actors = []

        for i in range(self.num_envs):
            object_idx_this_env = i % len(object_idx_list)
            self.obj_actors.append([])
            # create env instance
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)

            # compute aggregate size
            max_agg_bodies = self.num_shadow_hand_bodies + self.num_object_bodies_list[object_idx_this_env] + 1 #(table)
            max_agg_shapes = self.num_shadow_hand_shapes + self.num_object_shapes_list[object_idx_this_env] + 1
            if self.aggregate_mode >= 1:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            shadow_hand_start_pose = gymapi.Transform()
            # shadow_hand_start_pose.p = gymapi.Vec3(-0.37, -0.02, self.table_dims.z + self.hand_init_height_dict[object_idx_this_env])  # gymapi.Vec3(0.1, 0.1, 0.65)
            shadow_hand_start_pose.p = gymapi.Vec3(self.hand_bias[0], self.hand_bias[1],self.table_dims.z + self.hand_init_height_dict[object_idx_this_env] + self.hand_bias[2])
            shadow_hand_start_pose.r = gymapi.Quat().from_euler_zyx(0, 0, 0)  # gymapi.Quat().from_euler_zyx(0, -1.57, 0)

            object_start_pose = gymapi.Transform()
            self.object_rise = self.object_init_height_dict[object_idx_this_env]
            object_start_pose.p = gymapi.Vec3(0.0, 0.0, self.table_dims.z + self.object_rise)  # gymapi.Vec3(0.0, 0.0, 0.72)
            object_start_pose.r = gymapi.Quat().from_euler_zyx(0, 0, 0)  # gymapi.Quat().from_euler_zyx(1.57, 0, 0)

            table_pose = gymapi.Transform()
            table_pose.p = gymapi.Vec3(0.0, 0.0, 0.5 * self.table_dims.z)
            table_pose.r = gymapi.Quat().from_euler_zyx(-0., 0, 0)

            # add hand
            shadow_hand_actor = self._load_shadow_hand(env_ptr, i, self.shadow_hand_asset, self.shadow_hand_dof_props,
                                                       shadow_hand_start_pose)
            self.hand_start_states.append(
                [shadow_hand_start_pose.p.x, shadow_hand_start_pose.p.y, shadow_hand_start_pose.p.z,
                 shadow_hand_start_pose.r.x, shadow_hand_start_pose.r.y, shadow_hand_start_pose.r.z,
                 shadow_hand_start_pose.r.w,
                 0, 0, 0, 0, 0, 0])
            self.shadow_hands.append(shadow_hand_actor)

            # add object
            object_actor = self._load_object(env_ptr, i, self.object_asset_dict[object_idx_this_env],
                                             object_start_pose, 1.0)

            # set color
            color_cap = gymapi.Vec3(0.3, 0.6, 0.6)
            color_body = gymapi.Vec3(0.6, 0.2, 0.2)
            for o in range(self.num_object_bodies_list[i % len(self.env_dict)]):
                if o < self.num_object_bodies_list[i % len(self.env_dict)] - 1:
                    self.gym.set_rigid_body_color(env_ptr, object_actor, o, gymapi.MESH_VISUAL, color_body)
                else:
                    self.gym.set_rigid_body_color(env_ptr, object_actor, o, gymapi.MESH_VISUAL, color_cap)

            self.object_init_state.append([object_start_pose.p.x, object_start_pose.p.y, object_start_pose.p.z,
                                           object_start_pose.r.x, object_start_pose.r.y, object_start_pose.r.z,
                                           object_start_pose.r.w,
                                           0, 0, 0, 0, 0, 0])

            object_dof_props = self.gym.get_actor_dof_properties(env_ptr, object_actor)
            for object_dof_prop in object_dof_props:
                # print(object_dof_prop)
                # object_dof_prop[6] = 0.1
                object_dof_prop["damping"] = 0.1
                # object_dof_prop["velocity"] = 10.0
            self.gym.set_actor_dof_properties(env_ptr, object_actor, object_dof_props)
            # set friction
            # object_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, object_actor)
            # for object_shape_prop in object_shape_props:
            #     object_shape_prop.friction = 1
            # self.gym.set_actor_rigid_shape_properties(env_ptr, object_actor, object_shape_props)
            # object_body_props = self.gym.get_actor_rigid_body_properties(env_ptr, object_actor)
            # object_body_props[1].mass = 0.05
            # for object_body_prop in object_body_props:
            #     object_body_prop.mass = 1
            # self.gym.set_actor_rigid_body_properties(env_ptr, object_actor, object_body_props)
            self.obj_actors[i].append(object_actor)

            # add table
            table_handle = self.gym.create_actor(env_ptr, self.table_asset, table_pose, "table", i, -1, 0)
            # self.gym.set_rigid_body_texture(env_ptr, table_handle, 0, gymapi.MESH_VISUAL, table_texture_handle)
            table_idx = self.gym.get_actor_index(env_ptr, table_handle, gymapi.DOMAIN_SIM)
            self.table_indices.append(table_idx)

            # set friction
            table_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, table_handle)
            table_shape_props[0].friction = 1
            self.gym.set_actor_rigid_shape_properties(env_ptr, table_handle, table_shape_props)

            # Vision
            self._load_cameras(env_ptr, i, self.camera_props, self.camera_eye_list, self.camera_lookat_list)

            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            self.envs.append(env_ptr)
        self.hand_contact_idx_one_env = self.hand_contact_idx[:20]
        self.object_init_state = to_torch(self.object_init_state, device=self.device, dtype=torch.float).view(self.num_envs, 13)
        self.hand_start_states = to_torch(self.hand_start_states, device=self.device).view(self.num_envs, 13)
        self.fingertip_handles = to_torch(self.fingertip_handles, dtype=torch.long, device=self.device)
        self.hand_indices = to_torch(self.hand_indices, dtype=torch.long, device=self.device)
        self.table_indices = to_torch(self.table_indices, dtype=torch.long, device=self.device)
        self.hand_contact_idx = to_torch(self.hand_contact_idx, dtype=torch.long, device=self.device)
        self.object_indices = to_torch(self.object_indices, device=self.device, dtype=torch.long)

    def compute_reward(self,actions=None):
        self.dof_pos = self.shadow_hand_dof_pos

        self.rew_buf[:], self.reset_buf[:], self.reset_goal_buf[:], self.progress_buf[:], self.successes[:] = self.compute_hand_reward(
            self.object_dof_pos.squeeze(-1), self.object_dof_vel.squeeze(-1), self.reset_buf,
            self.progress_buf, self.successes, self.consecutive_successes,
            self.max_episode_length, self.av_factor
        )
        idx = torch.where(self.reset_buf==1)[0]
        self.env_mean_successes[idx] = torch.hstack([self.env_mean_successes[idx, 1:], self.successes[idx].view(-1, 1)])
        self.extras['successes'] = self.successes
        self.extras['env_mean_successes'] = self.env_mean_successes.mean(dim=-1)
        self.extras['consecutive_successes'] = self.consecutive_successes


    def compute_hand_reward(self,
                            object_dof_pos, object_dof_vel, reset_buf, progress_buf, successes, consecutive_successes,
                            max_episode_length: float, av_factor: float):
        if self.success_type=="one":
            return self.compute_hand_reward_one(object_dof_pos, object_dof_vel, reset_buf,
                                                progress_buf, successes, consecutive_successes,
                                                max_episode_length, av_factor)
        elif self.success_type=="half":
            return self.compute_hand_reward_half(object_dof_pos, object_dof_vel, reset_buf,
                                                progress_buf, successes, consecutive_successes,
                                                max_episode_length, av_factor)
        else:
            raise AttributeError(f"success type {self.success_type} not defined")
    def compute_hand_reward_one(self,
                            object_dof_pos, object_dof_vel, reset_buf, progress_buf, successes, consecutive_successes,
                            max_episode_length: float, av_factor: float):
        # contact_num = torch.sum(self.sensor_obs, dim=1)
        a, b, c, d = self.reward_weight
        r1 = torch.min(object_dof_pos, torch.tensor(7.0))
        # r1 = object_dof_pos * 2 // 3.14
        r2 = torch.clamp(object_dof_vel, -10.0, 10.0)
        # print(r2.max())



        if self.enable_touch_reward:
            # contacts = torch.zeros(self.sensor_obs.shape[0])
            # contact_num = torch.sum(self.sensor_obs, dim=1)
            # reward += d * contact_num
            contacts = self.match_contacts2(self.min_contacts).to(torch.float)

        else:
            contacts = torch.ones(self.sensor_obs.shape[0])

        action_penalty = torch.sum(self.actions ** 2, dim=-1)
        # action_penalty1 = torch.sum((self.actions-self.last_actions) ** 2, dim=-1)
        # self.action_penalty.append(action_penalty.mean().item())
        # self.action_penalty1.append(action_penalty1.mean().item())
        if self.enable_soft_contact:
            assert self.enable_touch_reward
            # soft contacts
            self.soft_contact = torch.cat([self.soft_contact, contacts.view(-1, 1)], dim=-1)[:, 1:]
            contacts = self.soft_contact.mean(dim=1)
            r2 = torch.where(r2 > 0, r2 * contacts, r2)
        else:
            r2 = torch.where(r2 > 0, torch.where(contacts > 0, r2, torch.zeros_like(r2)), r2)

        reward = a * r1 + b * r2 + d * action_penalty
        # reward = torch.where(r1 > 3.14 * 1.5, reward + 1, torch.where(r1 > 3.14, reward + 0.5, reward))


        if self.enable_distance_reward:
            fingertips = torch.stack([self.right_hand_ff_pos, self.right_hand_mf_pos, self.right_hand_rf_pos, self.right_hand_lf_pos, self.right_hand_th_pos], dim=1)
            fingertips_z = fingertips[..., -1] - 0.6  # table height 0.6
            target_dim_z = to_torch(list(self.hand_init_height_dict.values()),
                                    device=self.device) - 0.05 - 0.02  # a - 0.05 == the top of obj
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


        # print(f"r_p: {r1[0]}, r_v{r2[0]}, r_d{r_d[0]}")
        # reward 1:
        # reward = r2
        # reward = torch.where(object_dof_pos > 3.14 * 2, reward + 6, torch.where(object_dof_pos > 3.14, reward + 3, reward))

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
        reward = torch.where(successes>0, reward+5, reward)
        # reward *= 0.1  # scale reward
        # num_resets = torch.sum(resets)
        # cons_successes = torch.where(resets > 0, successes * resets, consecutive_successes)
        return reward, resets, goal_resets, progress_buf, successes#, cons_successes
    def compute_hand_reward_half(self,
                            object_dof_pos, object_dof_vel, reset_buf, progress_buf, successes, consecutive_successes,
                            max_episode_length: float, av_factor: float):
        a, b, c, d = self.reward_weight
        r1 = object_dof_pos
        r2 = torch.clamp((object_dof_pos - self.last_qpos) / self.dt, -10.0, 10.0)
        self.last_qpos = object_dof_pos.clone()
        if self.last_qpos_init==None:
            self.last_qpos_init = object_dof_pos.clone()
        # delta_qpos = r1[0] - self.last_qpos
        # print(f"r_p: {r1[0]}, r_v: {object_dof_vel[0]}, v_:{r2[0].item()}")
        # self.last_qpos = object_dof_pos[0].clone()
        contact = torch.any(self.sensor_obs, dim=-1)
        # if v > 0:
        #     if contact:
        #         v
        #     else:
        #         0
        # else:
        #     v
        r2 = torch.where((r2 > 0) & (contact==False), contact*r2, r2)

        reward = a * r1 + b * r2
        reward = torch.where(r1 > 3.14/2, reward + 5, torch.where(r1 > 3.14/4, reward + 2, reward))

        if self.enable_distance_reward:
            fingertips = torch.stack([self.right_hand_ff_pos, self.right_hand_mf_pos, self.right_hand_rf_pos, self.right_hand_lf_pos, self.right_hand_th_pos], dim=1)
            fingertips_z = fingertips[..., -1] - 0.6 # table height 0.6
            target_dim_z = to_torch(list(self.hand_init_height_dict.values()), device=self.device) - 0.05 - 0.02 # a - 0.05 == the top of obj
            target_dim_z = target_dim_z.repeat(fingertips_z.shape[0]//target_dim_z.shape[0]).view(-1, 1)
            dis_z = torch.norm(target_dim_z-fingertips_z, p=1, dim=1)
            r_d = torch.exp(-10 * dis_z)  #0~1
            # r_d = -torch.log(4*dis_z+1)     #-0.5~0
            reward += c * r_d


        if self.enable_touch_reward and self.obs_type in ["VisTac", "TacOnly"]:
            contact_num = torch.sum(self.sensor_obs, dim=1)
            reward += d * contact_num

        resets = reset_buf

        # # Find out which envs hit the goal and update successes count
        resets = torch.where((r1 > 3.14) & (contact==True), torch.ones_like(resets), resets)
        resets = torch.where(progress_buf >= max_episode_length, torch.ones_like(resets), resets)
        # tanh 2
        # reward = torch.where((resets == 1) & (progress_buf < max_episode_length), reward + 100, reward)

        goal_resets = resets
        successes = torch.where(r1 > 3.14, torch.ones_like(successes), successes)
        # if any(successes):
        #     print("hahaha")
        # success bonus tanh3
        reward = torch.where(successes>0, reward+20, reward)
        # reward *= 0.1  # scale reward
        # num_resets = torch.sum(resets)
        # cons_successes = torch.where(resets > 0, successes * resets, consecutive_successes)
        return reward, resets, goal_resets, progress_buf, successes#, cons_successes
    
    def compute_observations(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)

        robot_state = self.compute_robot_state(full_obs=self.full_obs)
        # object_state = self.compute_object_state(set_goal=False)
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

    def reset_idx(self, env_ids, goal_env_ids):
        # randomization can happen only at reset time, since it can reset actor positions on GPU
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

        # generate random values
        rand_floats = torch_rand_float(-1.0, 1.0, (len(env_ids), self.num_shadow_hand_dofs * 2 + 3), device=self.device)

        # reset shadow hand
        hand_indices = self.hand_indices[env_ids].to(torch.int32)
        self._reset_hand(env_ids, rand_floats, self.hand_indices[env_ids])

        # reset object
        self._reset_object(env_ids, rand_floats, self.object_indices[env_ids])

        # reset table
        table_indices = self.table_indices[env_ids]
        self._reset_table(env_ids)

        all_dof_indices = torch.unique(
            torch.cat([hand_indices, self.object_indices[env_ids].view(-1)]).to(torch.int32))
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(all_dof_indices), len(all_dof_indices))

        self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self.dof_targets),
                                                        gymtorch.unwrap_tensor(all_dof_indices), len(all_dof_indices))

        all_indices = torch.unique(torch.cat([hand_indices,
                                              self.object_indices[env_ids].view(-1),
                                              table_indices]).to(torch.int32))

        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_state_tensor),
                                                     gymtorch.unwrap_tensor(all_indices), len(all_indices))

        # if self.random_time:
        #     self.random_time = False
        #     self.progress_buf[env_ids] = torch.randint(0, self.max_episode_length, (len(env_ids),), device=self.device)
        # else:
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self.successes[env_ids] = 0

    def _reset_object(self, env_ids, rand_floats, object_indices):

        self.object_dof_pos[env_ids, :] = self.object_default_dof_pos
        self.object_dof_vel[env_ids, :] = self.object_default_dof_vel

        self.hand_positions[object_indices, :] = self.object_init_state[env_ids, 0:3]
        self.hand_orientations[object_indices, :] = self.object_init_state[env_ids, 3:7]
        self.hand_linvels[object_indices, :] = 0
        self.hand_angvels[object_indices, :] = 0
        self.root_state_tensor[object_indices, :] = self.saved_root_tensor[object_indices]
        return

    def _reset_table(self, env_ids):
        indices = self.table_indices[env_ids]
        self.root_state_tensor[indices, :] = self.saved_root_tensor[indices]
        return
    def step(self, actions):
        if self.dr_randomizations.get('actions', None):
            actions = self.dr_randomizations['actions']['noise_lambda'](actions)

        # apply actions
        self.pre_physics_step(actions)

        # step physics and render each frame
        for i in range(self.control_freq_inv):
            self.render()
            self.gym.simulate(self.sim)

        # to fix!
        if self.device == 'cpu':
            self.gym.fetch_results(self.sim, True)

        # compute observations, rewards, resets, ...
        self.post_physics_step()

        self.last_actions = self.actions.clone()

        if self.dr_randomizations.get('observations', None):
            self.obs_states_buf[:, :self.base_obs_dim] = self.dr_randomizations['observations']['noise_lambda'](self.obs_states_buf[:, :self.base_obs_dim])
    def match_contacts(self):
        contact_in_envs = []
        # self.sensor_obs
        for env in self.envs:
            contacts_list = list(self.gym.get_env_rigid_contacts(env))
            contact_in_env = []
            for contact_idx in self.hand_contact_idx_one_env:
                find_contact = False
                for contact in contacts_list:
                    if [self.bottle_cap_idx, contact_idx] in [[contact[2], contact[3]], [contact[3], contact[2]]]:
                        contact_in_env.append(True)
                        find_contact =True
                        break
                if not find_contact:
                    contact_in_env.append(False)
            contact_in_envs.append(contact_in_env)
        return to_torch(contact_in_envs, dtype=torch.float, device=self.device)
    def match_contacts2(self, min_contacts=1):
        contact_pairs, _ = torch.cat([torch.tensor(self.hand_contact_idx_one_env).view(-1, 1),
                                   torch.tensor(self.bottle_cap_idx).repeat(
                                       self.hand_contact_idx_one_env.__len__()).view(-1, 1)], dim=-1).sort(dim=1, descending=True)

        num_contacts_in_envs = self.sensor_obs.sum(dim=1)
        contact_envs = num_contacts_in_envs >= min_contacts
        contact_in_envs = torch.zeros_like(self.sensor_obs).to(torch.bool)
        for i, contact_happen_env in enumerate(contact_envs):
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
    #
    #
    # def pre_physics_step(self, actions):
    #     env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
    #     goal_env_ids = self.reset_goal_buf.nonzero(as_tuple=False).squeeze(-1)
    #
    #     if len(env_ids) > 0:
    #         self.reset_idx(env_ids, goal_env_ids)
    #
    #     self.actions = actions.clone().to(self.device)
    #     if self.use_relative_control:
    #         targets = self.prev_targets[:, self.actuated_dof_indices] + self.shadow_hand_dof_speed_scale * self.dt * self.actions
    #         self.cur_targets[:, self.actuated_dof_indices] = tensor_clamp(targets,
    #                                                                       self.shadow_hand_dof_lower_limits[self.actuated_dof_indices], self.shadow_hand_dof_upper_limits[self.actuated_dof_indices])
    #     else:
    #         self.cur_targets[:, self.actuated_dof_indices] = scale(self.actions,
    #                                                                self.shadow_hand_dof_lower_limits[self.actuated_dof_indices], self.shadow_hand_dof_upper_limits[self.actuated_dof_indices])
    #         self.cur_targets[:, self.actuated_dof_indices] = self.act_moving_average * self.cur_targets[:,
    #                                                                                                     self.actuated_dof_indices] + (1.0 - self.act_moving_average) * self.prev_targets[:, self.actuated_dof_indices]
    #         self.cur_targets[:, self.actuated_dof_indices] = tensor_clamp(self.cur_targets[:, self.actuated_dof_indices],
    #                                                                       self.shadow_hand_dof_lower_limits[self.actuated_dof_indices], self.shadow_hand_dof_upper_limits[self.actuated_dof_indices])
    #
    #     self.prev_targets[:, self.actuated_dof_indices] = self.cur_targets[:, self.actuated_dof_indices]
    #     self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.cur_targets))
    def add_gaussian_noise(self, data, mean=0.0, std=1.0):
        noise = torch.randn_like(data) * std + mean
        noisy_data = data + noise
        return noisy_data

    def hysteresis_thresholding(self, noisy_data, low_threshold=0.01, high_threshold=0.5):
        high_mask = noisy_data > high_threshold
        low_mask = (noisy_data > low_threshold) & (noisy_data <= high_threshold)
        output = torch.zeros_like(noisy_data, dtype=torch.float, device=self.device)


        for i in range(noisy_data.shape[1]):
            output[:, i] = high_mask[:, i]
            for j in range(1, noisy_data.shape[0]):
                if high_mask[j, i]:
                    output[j, i] = 1.0
                elif low_mask[j, i] and output[j - 1, i]:
                    output[j, i] = 1.0

        return output
    def compute_sensor_obs(self):
        # forces and torques
        contact = self.contact_force[self.hand_contact_idx].view(self.num_envs, self.num_force_sensors, 3)
        # vec_sensor = self.vec_sensor_tensor
        vec_sensor = contact
        vec_sensor = torch.norm(vec_sensor, p=2, dim=2)
        self.force = vec_sensor.cpu().numpy()
        if self.tac_noise is not None:
            vec_sensor = self.add_gaussian_noise(vec_sensor, std=self.tac_noise)
            if self.hysteresis:
                vec_sensor = self.hysteresis_thresholding(vec_sensor)
        self.sensor_obs = torch.zeros_like(vec_sensor)
        self.sensor_obs[vec_sensor > self.tactile_theshold] = 1
        # print(vec_sensor)
        return self.sensor_obs
    def apply_randomizations(self, dr_params):


        # If we don't have a randomization frequency, randomize every step
        rand_freq = dr_params.get("frequency", 1)

        # First, determine what to randomize:
        #   - non-environment parameters when > frequency steps have passed since the last non-environment
        #   - physical environments in the reset buffer, which have exceeded the randomization frequency threshold
        #   - on the first call, randomize everything
        self.last_step = self.gym.get_frame_count(self.sim)
        if self.first_randomization:
            do_nonenv_randomize = True
            env_ids = list(range(self.num_envs))
        else:
            do_nonenv_randomize = (self.last_step - self.last_rand_step) >= rand_freq
            rand_envs = torch.where(self.randomize_buf >= rand_freq, torch.ones_like(self.randomize_buf),
                                    torch.zeros_like(self.randomize_buf))
            rand_envs = torch.logical_and(rand_envs, self.reset_buf)
            env_ids = torch.nonzero(rand_envs, as_tuple=False).squeeze(-1).tolist()
            self.randomize_buf[rand_envs] = 0

        if do_nonenv_randomize:
            self.last_rand_step = self.last_step

        param_setters_map = get_property_setter_map(self.gym)
        param_setter_defaults_map = get_default_setter_args(self.gym)
        param_getters_map = get_property_getter_map(self.gym)

        # On first iteration, check the number of buckets
        # if self.first_randomization:
        #     check_buckets(self.gym, self.envs, dr_params)

        for nonphysical_param in ["observations", "actions"]:
            if nonphysical_param in dr_params and do_nonenv_randomize:
                dist = dr_params[nonphysical_param]["distribution"]
                op_type = dr_params[nonphysical_param]["operation"]
                sched_type = dr_params[nonphysical_param]["schedule"] if "schedule" in dr_params[nonphysical_param] else None
                sched_step = dr_params[nonphysical_param]["schedule_steps"] if "schedule" in dr_params[
                    nonphysical_param] else None
                op = operator.add if op_type == 'additive' else operator.mul

                if sched_type == 'linear':
                    sched_scaling = 1.0 / sched_step * \
                                    min(self.last_step, sched_step)
                elif sched_type == 'constant':
                    sched_scaling = 0 if self.last_step < sched_step else 1
                else:
                    sched_scaling = 1

                if dist == 'gaussian':
                    mu, var = dr_params[nonphysical_param]["range"]
                    mu_corr, var_corr = dr_params[nonphysical_param].get("range_correlated", [0., 0.])

                    if op_type == 'additive':
                        mu *= sched_scaling
                        var *= sched_scaling
                        mu_corr *= sched_scaling
                        var_corr *= sched_scaling
                    elif op_type == 'scaling':
                        var = var * sched_scaling  # scale up var over time
                        mu = mu * sched_scaling + 1.0 * \
                             (1.0 - sched_scaling)  # linearly interpolate

                        var_corr = var_corr * sched_scaling  # scale up var over time
                        mu_corr = mu_corr * sched_scaling + 1.0 * \
                                  (1.0 - sched_scaling)  # linearly interpolate


                    def noise_lambda(tensor, param_name=nonphysical_param):
                        params = self.dr_randomizations[param_name]
                        corr = params.get('corr', None)
                        if corr is None:
                            corr = torch.randn_like(tensor)
                            params['corr'] = corr
                        corr = corr * params['var_corr'] + params['mu_corr']
                        return op(
                            tensor, corr + torch.randn_like(tensor) * params['var'] + params['mu'])


                    self.dr_randomizations[nonphysical_param] = {'mu': mu, 'var': var, 'mu_corr': mu_corr, 'var_corr': var_corr,
                                                                 'noise_lambda': noise_lambda}

                elif dist == 'uniform':
                    lo, hi = dr_params[nonphysical_param]["range"]
                    lo_corr, hi_corr = dr_params[nonphysical_param].get("range_correlated", [0., 0.])

                    if op_type == 'additive':
                        lo *= sched_scaling
                        hi *= sched_scaling
                        lo_corr *= sched_scaling
                        hi_corr *= sched_scaling
                    elif op_type == 'scaling':
                        lo = lo * sched_scaling + 1.0 * (1.0 - sched_scaling)
                        hi = hi * sched_scaling + 1.0 * (1.0 - sched_scaling)
                        lo_corr = lo_corr * sched_scaling + 1.0 * (1.0 - sched_scaling)
                        hi_corr = hi_corr * sched_scaling + 1.0 * (1.0 - sched_scaling)


                    def noise_lambda(tensor, param_name=nonphysical_param):
                        params = self.dr_randomizations[param_name]
                        corr = params.get('corr', None)
                        if corr is None:
                            corr = torch.randn_like(tensor)
                            params['corr'] = corr
                        corr = corr * (params['hi_corr'] - params['lo_corr']) + params['lo_corr']
                        return op(tensor, corr + torch.rand_like(tensor) * (params['hi'] - params['lo']) + params['lo'])


                    self.dr_randomizations[nonphysical_param] = {'lo': lo, 'hi': hi, 'lo_corr': lo_corr, 'hi_corr': hi_corr,
                                                                 'noise_lambda': noise_lambda}

        # if "sim_params" in dr_params and do_nonenv_randomize:
        #     prop_attrs = dr_params["sim_params"]
        #     prop = self.gym.get_sim_params(self.sim)
        #
        #     if self.first_randomization:
        #         self.original_props["sim_params"] = {
        #             attr: getattr(prop, attr) for attr in dir(prop)}
        #
        #     for attr, attr_randomization_params in prop_attrs.items():
        #         apply_random_samples(
        #             prop, self.original_props["sim_params"], attr, attr_randomization_params, self.last_step)
        #
        #     self.gym.set_sim_params(self.sim, prop)

        # If self.actor_params_generator is initialized: use it to
        # sample actor simulation params. This gives users the
        # freedom to generate samples from arbitrary distributions,
        # e.g. use full-covariance distributions instead of the DR's
        # default of treating each simulation parameter independently.
        # extern_offsets = {}
        # if self.actor_params_generator is not None:
        #     for env_id in env_ids:
        #         self.extern_actor_params[env_id] = \
        #             self.actor_params_generator.sample()
        #         extern_offsets[env_id] = 0
        #
        # for actor, actor_properties in dr_params["actor_params"].items():
        #     for env_id in env_ids:
        #         env = self.envs[env_id]
        #         handle = self.gym.find_actor_handle(env, actor)
        #         extern_sample = self.extern_actor_params[env_id]
        #
        #         for prop_name, prop_attrs in actor_properties.items():
        #             if prop_name == 'color':
        #                 num_bodies = self.gym.get_actor_rigid_body_count(
        #                     env, handle)
        #                 for n in range(num_bodies):
        #                     self.gym.set_rigid_body_color(env, handle, n, gymapi.MESH_VISUAL,
        #                                                   gymapi.Vec3(random.uniform(0, 1), random.uniform(0, 1),
        #                                                               random.uniform(0, 1)))
        #                 continue
        #             if prop_name == 'scale':
        #                 attr_randomization_params = prop_attrs
        #                 sample = generate_random_samples(attr_randomization_params, 1,
        #                                                  self.last_step, None)
        #                 og_scale = 1
        #                 if attr_randomization_params['operation'] == 'scaling':
        #                     new_scale = og_scale * sample
        #                 elif attr_randomization_params['operation'] == 'additive':
        #                     new_scale = og_scale + sample
        #                 self.gym.set_actor_scale(env, handle, new_scale)
        #                 continue
        #
        #             prop = param_getters_map[prop_name](env, handle)
        #             if isinstance(prop, list):
        #                 if self.first_randomization:
        #                     self.original_props[prop_name] = [
        #                         {attr: getattr(p, attr) for attr in dir(p)} for p in prop]
        #                 for p, og_p in zip(prop, self.original_props[prop_name]):
        #                     for attr, attr_randomization_params in prop_attrs.items():
        #                         smpl = None
        #                         if self.actor_params_generator is not None:
        #                             smpl, extern_offsets[env_id] = get_attr_val_from_sample(
        #                                 extern_sample, extern_offsets[env_id], p, attr)
        #                         apply_random_samples(
        #                             p, og_p, attr, attr_randomization_params,
        #                             self.last_step, smpl)
        #             else:
        #                 if self.first_randomization:
        #                     self.original_props[prop_name] = deepcopy(prop)
        #                 for attr, attr_randomization_params in prop_attrs.items():
        #                     smpl = None
        #                     if self.actor_params_generator is not None:
        #                         smpl, extern_offsets[env_id] = get_attr_val_from_sample(
        #                             extern_sample, extern_offsets[env_id], prop, attr)
        #                     apply_random_samples(
        #                         prop, self.original_props[prop_name], attr,
        #                         attr_randomization_params, self.last_step, smpl)
        #
        #             setter = param_setters_map[prop_name]
        #             default_args = param_setter_defaults_map[prop_name]
        #             setter(env, handle, prop, *default_args)

        # if self.actor_params_generator is not None:
        #     for env_id in env_ids:  # check that we used all dims in sample
        #         if extern_offsets[env_id] > 0:
        #             extern_sample = self.extern_actor_params[env_id]
        #             if extern_offsets[env_id] != extern_sample.shape[0]:
        #                 print('env_id', env_id,
        #                       'extern_offset', extern_offsets[env_id],
        #                       'vs extern_sample.shape', extern_sample.shape)
        #                 raise Exception("Invalid extern_sample size")

        self.first_randomization = False