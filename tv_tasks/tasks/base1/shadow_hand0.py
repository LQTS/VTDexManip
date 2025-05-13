# Copyright (c) 2018-2023, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np
import os
import torch

from tv_tasks.utils.torch_jit_utils import *
from tv_tasks.tasks.base.base_task import BaseTask

from isaacgym import gymtorch
from isaacgym import gymapi
import pickle
import random


class ShadowHandBase(BaseTask):

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):

        self.cfg = cfg
        self.sim_params = sim_params
        self.physics_engine = physics_engine
        self.randomize = self.cfg["task"]["randomize"]
        self.randomization_params = self.cfg["task"]["randomization_params"]
        self.aggregate_mode = self.cfg["env"]["aggregateMode"]

        self.dist_reward_scale = self.cfg["env"]["distRewardScale"]
        self.rot_reward_scale = self.cfg["env"]["rotRewardScale"]
        self.action_penalty_scale = self.cfg["env"]["actionPenaltyScale"]
        self.success_tolerance = self.cfg["env"]["successTolerance"]
        self.reach_goal_bonus = self.cfg["env"]["reachGoalBonus"]
        self.fall_dist = self.cfg["env"]["fallDistance"]
        self.fall_penalty = self.cfg["env"]["fallPenalty"]
        self.rot_eps = self.cfg["env"]["rotEps"]

        self.vel_obs_scale = 0.2  # scale factor of velocity based observations
        self.force_torque_obs_scale = 10.0  # scale factor of velocity based observations

        self.reset_position_noise = self.cfg["env"]["resetPositionNoise"]
        self.reset_rotation_noise = self.cfg["env"]["resetRotationNoise"]
        self.reset_dof_pos_noise = self.cfg["env"]["resetDofPosRandomInterval"]
        self.reset_dof_vel_noise = self.cfg["env"]["resetDofVelRandomInterval"]
        self.transition_scale = cfg["env"]["transition_scale"]
        self.orientation_scale = cfg["env"]["orientation_scale"]

        self.force_scale = self.cfg["env"].get("forceScale", 0.0)
        self.force_prob_range = self.cfg["env"].get("forceProbRange", [0.001, 0.1])
        self.force_decay = self.cfg["env"].get("forceDecay", 0.99)
        self.force_decay_interval = self.cfg["env"].get("forceDecayInterval", 0.08)

        self.shadow_hand_dof_speed_scale = self.cfg["env"]["dofSpeedScale"]
        self.use_relative_control = self.cfg["env"]["useRelativeControl"]
        self.act_moving_average = self.cfg["env"]["actionsMovingAverage"]

        self.debug_viz = self.cfg["env"]["enableDebugVis"]

        self.max_episode_length = self.cfg["env"]["episodeLength"]
        self.reset_time = self.cfg["env"].get("resetTime", -1.0)
        self.print_success_stat = self.cfg["env"]["printNumSuccesses"]
        self.max_consecutive_successes = self.cfg["env"]["maxConsecutiveSuccesses"]
        self.av_factor = self.cfg["env"].get("averFactor", 0.1)

        self.object_type = self.cfg["env"]["objectType"]
        assert self.object_type in ["block", "egg", "pen"]

        self.ignore_z = (self.object_type == "pen")

        self.asset_files_dict = {
            "block": "urdf/objects/cube_multicolor.urdf",
            "egg": "mjcf/open_ai_assets/hand/egg.xml",
            "pen": "mjcf/open_ai_assets/hand/pen.xml"
        }

        if "asset" in self.cfg["env"]:
            self.asset_files_dict["block"] = self.cfg["env"]["asset"].get("assetFileNameBlock", self.asset_files_dict["block"])
            self.asset_files_dict["egg"] = self.cfg["env"]["asset"].get("assetFileNameEgg", self.asset_files_dict["egg"])
            self.asset_files_dict["pen"] = self.cfg["env"]["asset"].get("assetFileNamePen", self.asset_files_dict["pen"])

        # can be "openai", "full_no_vel", "full", "full_state"
        self.obs_type = self.cfg["env"]["obs_type"]

        # if not (self.obs_type in ["openai", "full_no_vel", "full", "full_state"]):
        #     raise Exception(
        #         "Unknown type of observations!\nobservationType should be one of: [openai, full_no_vel, full, full_state]")

        print("Obs type:", self.obs_type)

        self.num_obs_dict = {
            "openai": 42,
            "full_no_vel": 77,
            "full": 157,
            "full_state": 211
        }

        self.up_axis = 'z'
        self.fixed_hand = self.cfg['env'].get('fixed_hand', True)


        self.fingertips = ["robot0:ffdistal", "robot0:mfdistal", "robot0:rfdistal", "robot0:lfdistal", "robot0:thdistal"]
        self.num_fingertips = len(self.fingertips)

        self.force_sensor_body = [
            "fs_lfdistal", "fs_rfdistal", "fs_mfdistal", "fs_ffdistal","fs_thdistal",
            "fs_lfmiddle", "fs_rfmiddle", "fs_mfmiddle", "fs_ffmiddle","fs_thmiddle",
            "fs_lfproximal", "fs_rfproximal", "fs_mfproximal", "fs_ffproximal","fs_thproximal",
            'fs_lfknuckle', 'fs_rfknuckle', 'fs_mfknuckle', 'fs_ffknuckle','fs_palm'
        ]
        self.num_force_sensors = len(self.force_sensor_body)

        self.use_vel_obs = False
        self.fingertip_obs = True
        self.asymmetric_obs = self.cfg["env"]["asymmetric_observations"]

        num_states = 0
        if self.asymmetric_obs:
            num_states = 211

        # self.cfg["env"]["numObservations"] = self.num_obs_dict[self.obs_type]
        self.cfg["env"]["numStates"] = num_states
        self.cfg["env"]["numActions"] = 26
        self.cfg["device_type"] = device_type
        self.cfg["device_id"] = device_id
        self.cfg["headless"] = headless

        self.cfg["env"]["img_dim"] = (self.cam_h, self.cam_w, 3)


        # # Vision
        # self.cam_w = self.cfg["env"]['img_w']
        # self.cam_h = self.cfg["env"]['img_h']
        # self.cfg["env"]["img_dim"] = (self.cam_h, self.cam_w, 3)
        # self.segmentation_id = {
        #     'hand': 2,
        #     'object': 3,
        #     'goal': 4,
        #     'table': 1,
        # }
        # self.camera_depth_tensor_list = []
        # self.camera_rgb_tensor_list = []
        # self.camera_seg_tensor_list = []
        # self.camera_vinv_mat_list = []
        # self.camera_proj_mat_list = []
        # self.camera_handles = []
        # self.num_cameras = 1
        # self._cfg_camera_props()
        # self._cfg_camera_pose()

        self.num_envs = self.cfg['env']['numEnvs']
        self.env_origin = torch.zeros((self.num_envs, 3), dtype=torch.float)

        super().__init__(cfg=self.cfg, enable_camera_sensors=True)

        self.dt = self.sim_params.dt
        control_freq_inv = self.cfg["env"].get("controlFrequencyInv", 1)
        if self.reset_time > 0.0:
            self.max_episode_length = int(round(self.reset_time/(control_freq_inv * self.dt)))
            print("Reset time: ", self.reset_time)
            print("New episode length: ", self.max_episode_length)

        if self.viewer != None:
            cam_pos = gymapi.Vec3(10.0, 5.0, 1.0)
            cam_target = gymapi.Vec3(6.0, 5.0, 0.0)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        # get gym GPU state tensors
        self.env_origin = to_torch(self.env_origin, device=self.device)

        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        contact_force = self.gym.acquire_net_contact_force_tensor(self.sim)

        if self.obs_type == "Base" or self.asymmetric_obs:
            sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
            self.vec_sensor_tensor = gymtorch.wrap_tensor(sensor_tensor).view(self.num_envs, self.num_fingertips * 6)

        dof_force_tensor = self.gym.acquire_dof_force_tensor(self.sim)
        self.dof_force_tensor = gymtorch.wrap_tensor(dof_force_tensor).view(self.num_envs, self.num_shadow_hand_dofs + self.num_object_dofs)

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)


        # create some wrapper tensors for different slices\
        self.contact_force = gymtorch.wrap_tensor(contact_force)
        self.shadow_hand_default_dof_pos = torch.zeros(self.num_shadow_hand_dofs, dtype=torch.float, device=self.device)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.shadow_hand_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, :self.num_shadow_hand_dofs]
        self.shadow_hand_dof_pos = self.shadow_hand_dof_state[..., 0]
        self.shadow_hand_dof_vel = self.shadow_hand_dof_state[..., 1]

        self.object_default_dof_pos = torch.zeros(self.num_object_dofs, dtype=torch.float, device=self.device)
        self.object_default_dof_vel = torch.zeros(self.num_object_dofs, dtype=torch.float, device=self.device)
        self.object_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, self.num_shadow_hand_dofs::]
        self.object_dof_pos = self.object_dof_state[..., 0]
        self.object_dof_vel = self.object_dof_state[..., 1]

        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor)
        self.num_bodies = self.rigid_body_states.view(self.num_envs, -1, 13).shape[1]
        self.apply_forces = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device, dtype=torch.float)
        self.apply_torque = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device, dtype=torch.float)


        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(-1, 13)
        self.hand_positions = self.root_state_tensor[:, 0:3]
        self.hand_orientations = self.root_state_tensor[:, 3:7]
        self.hand_linvels = self.root_state_tensor[:, 7:10]
        self.hand_angvels = self.root_state_tensor[:, 10:13]
        self.saved_root_tensor = self.root_state_tensor.clone()

        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs
        self.prev_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self.cur_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self.dof_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)


        self.global_indices = torch.arange(self.num_envs * 3, dtype=torch.int32, device=self.device).view(self.num_envs, -1)
        self.x_unit_tensor = to_torch([1, 0, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.y_unit_tensor = to_torch([0, 1, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.z_unit_tensor = to_torch([0, 0, 1], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))

        self.reset_goal_buf = self.reset_buf.clone()
        self.successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.consecutive_successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        self.av_factor = to_torch(self.av_factor, dtype=torch.float, device=self.device)

        self.total_successes = 0
        self.total_resets = 0

        # object apply random forces parameters
        self.force_decay = to_torch(self.force_decay, dtype=torch.float, device=self.device)
        self.force_prob_range = to_torch(self.force_prob_range, dtype=torch.float, device=self.device)
        self.random_force_prob = torch.exp((torch.log(self.force_prob_range[0]) - torch.log(self.force_prob_range[1]))
                                           * torch.rand(self.num_envs, device=self.device) + torch.log(self.force_prob_range[1]))

        # self.rb_forces = torch.zeros((self.num_envs, self.num_bodies, 3), dtype=torch.float, device=self.device)

    def create_sim(self):
        self.dt = self.sim_params.dt
        self.up_axis_idx = self.set_sim_params_up_axis(self.sim_params, self.up_axis)
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

        # If randomizing, apply once immediately on startup before the fist sim step
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_hand_asset(self):
        # Retrieve asset paths
        self.asset_root = self.cfg["env"]["asset"]["assetRoot"]
        shadow_hand_asset_file = self.cfg["env"]["asset"]["assetFileNameRobot"]

        self.hand_asset_options = gymapi.AssetOptions()
        self.hand_asset_options.flip_visual_attachments = False
        self.hand_asset_options.fix_base_link = self.fixed_hand
        self.hand_asset_options.collapse_fixed_joints = True
        self.hand_asset_options.disable_gravity = True
        self.hand_asset_options.thickness = 0.001
        self.hand_asset_options.angular_damping = 0.01
        # self.hand_asset_options.linear_damping = 0.01

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
        if self.obs_type == "Base" or self.asymmetric_obs:
            sensor_pose = gymapi.Transform()
            for ft_handle in self.fingertip_handles:
                self.gym.create_asset_force_sensor(self.shadow_hand_asset, ft_handle, sensor_pose)

    # def _create_obj_asset(self):
    #     # load manipulated object and goal assets
    #     object_asset_file = self.asset_files_dict[self.object_type]
    #     self.object_asset_options = gymapi.AssetOptions()
    #     self.object_asset = self.gym.load_asset(self.sim, self.asset_root, object_asset_file, self.object_asset_options)
    #     self.object_asset_options.disable_gravity = True
    #     self.object_rb_count = self.gym.get_asset_rigid_body_count(self.object_asset)
    #
    #     self.goal_asset = self.gym.load_asset(self.sim, self.asset_root, object_asset_file, self.object_asset_options)

    def _create_obj_asset(self):
        self.obj_asset_root = self.asset_root + self.cfg["env"]["asset"]["assetFileNameObj"]
        self.env_dict = self.cfg['env']['env_dict']
        # Retrieve asset paths
        obj_height_file = self.obj_asset_root + '/urdf/obj_info/obj_init_height.pickle'
        hand_height_file = self.obj_asset_root + '/urdf/obj_info/hand_init_height.pickle'
        with open(obj_height_file, "rb") as fh:
            self.object_init_height = pickle.load(fh)
        with open(hand_height_file, "rb") as fh:
            self.hand_init_height = pickle.load(fh)

        self.object_idx = []
        self.num_object_bodies_list = []
        self.num_object_shapes_list = []

        self.object_init_height_dict = {}
        self.hand_init_height_dict = {}
        self.object_asset_dict = {}
        self.goal_asset_dict = {}

        for object_id, object_code in enumerate(self.env_dict):
            # load manipulated object and goal assets
            self.object_asset_options = gymapi.AssetOptions()
            self.object_asset_options.density = 500
            self.object_asset_options.fix_base_link = True
            self.object_asset_options.use_mesh_materials = True
            self.object_asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
            self.object_asset_options.override_com = True
            self.object_asset_options.override_inertia = True
            # object_asset_options.vhacd_enabled = True
            # object_asset_options.vhacd_params = gymapi.VhacdParams()
            # object_asset_options.vhacd_params.resolution = 300000
            self.object_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE

            self.object_asset_options.disable_gravity = True

            object_asset_file = object_code + ".urdf"
            object_asset = self.gym.load_asset(self.sim, self.obj_asset_root + "/urdf/mesh/" + object_code,
                                                      object_asset_file, self.object_asset_options)

            goal_asset = self.gym.load_asset(self.sim, self.obj_asset_root + "/urdf/mesh/" + object_code,
                                             object_asset_file, self.object_asset_options)

            self.object_asset_dict[object_id] = object_asset
            self.goal_asset_dict[object_id] = goal_asset

            self.object_init_height_dict[object_id] = self.object_init_height[object_code]
            self.hand_init_height_dict[object_id] = self.hand_init_height[object_code]
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

    def _create_table_asset(self, table_dims):
        self.table_asset_options = gymapi.AssetOptions()
        self.table_asset_options.fix_base_link = True
        self.table_asset_options.flip_visual_attachments = True
        self.table_asset_options.collapse_fixed_joints = True
        self.table_asset_options.disable_gravity = False
        self.table_asset_options.thickness = 0.001
        self.table_asset = self.gym.create_box(self.sim, table_dims.x, table_dims.y, table_dims.z,
                                          gymapi.AssetOptions())

    def envs_config(self):
        self.shadow_hands = []
        self.envs = []

        self.object_init_state = []
        self.hand_start_states = []

        self.hand_indices = []
        self.fingertip_indices = []
        self.object_indices = []
        self.table_indices = []
        self.goal_object_indices = []
        self.hand_contact_idx = []

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)
        # load shadow hand_ asset
        self._create_hand_asset()
        self._create_obj_asset()

        shadow_hand_start_pose = gymapi.Transform()
        shadow_hand_start_pose.p = gymapi.Vec3(*get_axis_params(0.5, self.up_axis_idx))

        object_start_pose = gymapi.Transform()
        object_start_pose.p = gymapi.Vec3()
        object_start_pose.p.x = shadow_hand_start_pose.p.x
        pose_dy, pose_dz = -0.39, 0.10

        object_start_pose.p.y = shadow_hand_start_pose.p.y + pose_dy
        object_start_pose.p.z = shadow_hand_start_pose.p.z + pose_dz

        if self.object_type == "pen":
            object_start_pose.p.z = shadow_hand_start_pose.p.z + 0.02

        self.goal_displacement = gymapi.Vec3(-0.2, -0.06, 0.12)
        self.goal_displacement_tensor = to_torch(
            [self.goal_displacement.x, self.goal_displacement.y, self.goal_displacement.z], device=self.device)
        goal_start_pose = gymapi.Transform()
        goal_start_pose.p = object_start_pose.p + self.goal_displacement

        goal_start_pose.p.z -= 0.04

        # compute aggregate size
        max_agg_bodies = self.num_shadow_hand_bodies + 2
        max_agg_shapes = self.num_shadow_hand_shapes + 2

        self.envs_config()

        self.object_rb_handles = list(range(self.shadow_hand_rb_count, self.shadow_hand_rb_count + self.object_rb_count))

        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(
                self.sim, lower, upper, num_per_row
            )

            if self.aggregate_mode >= 1:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # add hand - collision filter = -1 to use asset collision filters set in mjcf loader
            shadow_hand_actor = self._load_shadow_hand(env_ptr, i, self.shadow_hand_asset, self.shadow_hand_dof_props,
                                                       shadow_hand_start_pose)
            self.hand_start_states.append([shadow_hand_start_pose.p.x, shadow_hand_start_pose.p.y, shadow_hand_start_pose.p.z,
                                           shadow_hand_start_pose.r.x, shadow_hand_start_pose.r.y, shadow_hand_start_pose.r.z, shadow_hand_start_pose.r.w,
                                           0, 0, 0, 0, 0, 0])

            # add object
            object_handle = self.gym.create_actor(env_ptr, self.object_asset, object_start_pose, "object", i, 0, 0)
            self.object_init_state.append([object_start_pose.p.x, object_start_pose.p.y, object_start_pose.p.z,
                                           object_start_pose.r.x, object_start_pose.r.y, object_start_pose.r.z, object_start_pose.r.w,
                                           0, 0, 0, 0, 0, 0])
            object_idx = self.gym.get_actor_index(env_ptr, object_handle, gymapi.DOMAIN_SIM)
            self.object_indices.append(object_idx)

            # add goal object
            goal_handle = self.gym.create_actor(env_ptr, self.goal_asset, goal_start_pose, "goal_object", i + self.num_envs, 0, 0)
            goal_object_idx = self.gym.get_actor_index(env_ptr, goal_handle, gymapi.DOMAIN_SIM)
            self.goal_object_indices.append(goal_object_idx)

            if self.object_type != "block":
                self.gym.set_rigid_body_color(
                    env_ptr, object_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.6, 0.72, 0.98))
                self.gym.set_rigid_body_color(
                    env_ptr, goal_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.6, 0.72, 0.98))

            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            self.envs.append(env_ptr)
            self.shadow_hands.append(shadow_hand_actor)

        # we are not using new mass values after DR when calculating random forces applied to an object,
        # which should be ok as long as the randomization range is not too big
        object_rb_props = self.gym.get_actor_rigid_body_properties(env_ptr, object_handle)
        self.object_rb_masses = [prop.mass for prop in object_rb_props]

        self.object_init_state = to_torch(self.object_init_state, device=self.device, dtype=torch.float).view(self.num_envs, 13)
        self.goal_states = self.object_init_state.clone()
        self.goal_states[:, self.up_axis_idx] -= 0.04
        self.goal_init_state = self.goal_states.clone()
        self.hand_start_states = to_torch(self.hand_start_states, device=self.device).view(self.num_envs, 13)

        self.fingertip_handles = to_torch(self.fingertip_handles, dtype=torch.long, device=self.device)
        self.object_rb_handles = to_torch(self.object_rb_handles, dtype=torch.long, device=self.device)
        self.object_rb_masses = to_torch(self.object_rb_masses, dtype=torch.float, device=self.device)

        self.hand_indices = to_torch(self.hand_indices, dtype=torch.long, device=self.device)
        self.object_indices = to_torch(self.object_indices, dtype=torch.long, device=self.device)
        self.goal_object_indices = to_torch(self.goal_object_indices, dtype=torch.long, device=self.device)

    def _load_shadow_hand(self, env_ptr, env_id, shadow_hand_asset, hand_dof_props, init_hand_actor_pose):

        hand_actor = self.gym.create_actor(env_ptr, shadow_hand_asset, init_hand_actor_pose, "hand", env_id, -1,
                                           self.segmentation_id['hand'])
        # if self.has_sensor:
        #     self.gym.enable_actor_dof_force_sensors(env_ptr, hand_actor)

        self.gym.set_actor_dof_properties(env_ptr, hand_actor, hand_dof_props)
        hand_idx = self.gym.get_actor_index(env_ptr, hand_actor, gymapi.DOMAIN_SIM)
        self.hand_indices.append(hand_idx)


        colorx = random.uniform(0, 1)
        colory = random.uniform(0, 1)
        colorz = random.uniform(0, 1)
        hand_color = gymapi.Vec3(colorx, colory, colorz)

        sensor_color = gymapi.Vec3(1, 0.16355, 0.16355)
        for o in range(self.num_shadow_hand_bodies):
            self.gym.set_rigid_body_color(env_ptr, hand_actor, o, gymapi.MESH_VISUAL, hand_color)
        for o in self.sensor_indices:
            self.gym.set_rigid_body_color(env_ptr, hand_actor, o, gymapi.MESH_VISUAL, sensor_color)

        for o in range(len(self.force_sensor_body)):
            force_sensor_handle = self.gym.find_actor_rigid_body_index(env_ptr, hand_actor,
                                                                       self.force_sensor_body[o], gymapi.DOMAIN_SIM)
            self.hand_contact_idx.append(force_sensor_handle)
            
        for o in range(len(self.fingertips)):
            fingertip_env_handle = self.gym.find_actor_rigid_body_index(env_ptr, hand_actor,
                                                                       self.fingertips[o], gymapi.DOMAIN_SIM)
            self.fingertip_indices.append(fingertip_env_handle)

        return hand_actor

    def _load_object(self, env_ptr, env_id, object_asset, init_object_pose, scale=1.0):

        object_actor = self.gym.create_actor(env_ptr, object_asset, init_object_pose, "object", env_id, 0,
                                             self.segmentation_id['object'])
        object_idx = self.gym.get_actor_index(env_ptr, object_actor, gymapi.DOMAIN_SIM)
        self.object_indices.append(object_idx)
        self.gym.set_actor_scale(env_ptr, object_actor, scale)

        return object_actor

    def _load_goal(self, env_ptr, env_id, goal_asset, init_goal_pose, scale=1.0):

        goal_actor = self.gym.create_actor(env_ptr, goal_asset, init_goal_pose, "goal_object", env_id + self.num_envs, 0, self.segmentation_id['goal'])
        goal_object_idx = self.gym.get_actor_index(env_ptr, goal_actor, gymapi.DOMAIN_SIM)
        self.goal_object_indices.append(goal_object_idx)
        self.gym.set_actor_scale(env_ptr, goal_actor, scale)

        return goal_actor

    def set_camera(self, cfg, height):
        # Vision
        self.cam_w = cfg["env"]['img_w']
        self.cam_h = cfg["env"]['img_h']
        self.segmentation_id = {
            'hand': 2,
            'object': 3,
            'goal': 4,
            'table': 1,
        }
        self.camera_depth_tensor_list = []
        self.camera_rgb_tensor_list = []
        self.camera_seg_tensor_list = []
        self.camera_vinv_mat_list = []
        self.camera_proj_mat_list = []
        self.camera_handles = []
        self.num_cameras = 1
        self._cfg_camera_props()
        self._cfg_camera_pose(cfg, height)

    def _cfg_camera_props(self):
        self.camera_props = gymapi.CameraProperties()
        self.camera_props.horizontal_fov = 45
        self.camera_props.width = self.cam_w
        self.camera_props.height = self.cam_h
        self.camera_props.enable_tensors = True
        return

    def _cfg_camera_pose(self, cfg, height=0):
        self.camera_eye_list = []
        self.camera_lookat_list = []
        camera_eye_list = cfg['env']['vision']['camera']['eye']
        camera_lookat_list = cfg['env']['vision']['camera']['lookat']
        centor = np.array([0.0, 0.0, height])
        for i in range(self.num_cameras):
            camera_eye = np.array(camera_eye_list[i]) + centor
            camera_lookat = np.array(camera_lookat_list[i]) + centor
            self.camera_eye_list.append(gymapi.Vec3(*list(camera_eye)))
            self.camera_lookat_list.append(gymapi.Vec3(*list(camera_lookat)))
        return

    def _load_cameras(self, env_ptr, env_id, camera_props, camera_eye_list, camera_lookat_list):
        camera_handles = []
        depth_tensors = []
        rgb_tensors = []
        seg_tensors = []
        vinv_mats = []
        proj_mats = []

        origin = self.gym.get_env_origin(env_ptr)
        self.env_origin[env_id][0] = origin.x
        self.env_origin[env_id][1] = origin.y
        self.env_origin[env_id][2] = origin.z

        for i in range(self.num_cameras):
            camera_handle = self.gym.create_camera_sensor(env_ptr, camera_props)

            camera_eye = camera_eye_list[i]
            camera_lookat = camera_lookat_list[i]
            self.gym.set_camera_location(camera_handle, env_ptr, camera_eye, camera_lookat)
            raw_depth_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, env_ptr, camera_handle,
                                                                    gymapi.IMAGE_DEPTH)
            depth_tensor = gymtorch.wrap_tensor(raw_depth_tensor)
            depth_tensors.append(depth_tensor)

            raw_rgb_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, env_ptr, camera_handle, gymapi.IMAGE_COLOR)
            rgb_tensor = gymtorch.wrap_tensor(raw_rgb_tensor)
            rgb_tensors.append(rgb_tensor)

            raw_seg_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, env_ptr, camera_handle,
                                                                  gymapi.IMAGE_SEGMENTATION)
            seg_tensor = gymtorch.wrap_tensor(raw_seg_tensor)
            seg_tensors.append(seg_tensor)

            vinv_mat = torch.inverse(
                (to_torch(self.gym.get_camera_view_matrix(self.sim, env_ptr, camera_handle), device=self.device)))
            vinv_mats.append(vinv_mat)

            proj_mat = to_torch(self.gym.get_camera_proj_matrix(self.sim, env_ptr, camera_handle), device=self.device)
            proj_mats.append(proj_mat)

            camera_handles.append(camera_handle)

        self.camera_depth_tensor_list.append(depth_tensors)
        self.camera_rgb_tensor_list.append(rgb_tensors)
        self.camera_seg_tensor_list.append(seg_tensors)
        self.camera_vinv_mat_list.append(vinv_mats)
        self.camera_proj_mat_list.append(proj_mats)

        return


    def compute_reward(self, actions):
        self.rew_buf[:], self.reset_buf[:], self.reset_goal_buf[:], self.progress_buf[:], self.successes[:], self.consecutive_successes[:] = compute_hand_reward(
            self.rew_buf, self.reset_buf, self.reset_goal_buf, self.progress_buf, self.successes, self.consecutive_successes,
            self.max_episode_length, self.object_pos, self.object_rot, self.goal_pos, self.goal_rot,
            self.dist_reward_scale, self.rot_reward_scale, self.rot_eps, self.actions, self.action_penalty_scale,
            self.success_tolerance, self.reach_goal_bonus, self.fall_dist, self.fall_penalty,
            self.max_consecutive_successes, self.av_factor, (self.object_type == "pen")
        )

        self.extras['consecutive_successes'] = self.consecutive_successes.mean()

        if self.print_success_stat:
            self.total_resets = self.total_resets + self.reset_buf.sum()
            direct_average_successes = self.total_successes + self.successes.sum()
            self.total_successes = self.total_successes + (self.successes * self.reset_buf).sum()

            # The direct average shows the overall result more quickly, but slightly undershoots long term
            # policy performance.
            print("Direct average consecutive successes = {:.1f}".format(direct_average_successes/(self.total_resets + self.num_envs)))
            if self.total_resets > 0:
                print("Post-Reset average consecutive successes = {:.1f}".format(self.total_successes/self.total_resets))

    def compute_observations(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)

        self.right_hand_pos = self.rigid_body_states.view(self.num_envs, -1, 13)[:, 3, 0:3]
        self.right_hand_rot = self.rigid_body_states.view(self.num_envs, -1, 13)[:, 3, 3:7]
        self.right_hand_pos = self.right_hand_pos + quat_apply(self.right_hand_rot,
                                                               to_torch([0, 0, 1], device=self.device).repeat(
                                                                   self.num_envs, 1) * 0.08)
        self.right_hand_pos = self.right_hand_pos + quat_apply(self.right_hand_rot,
                                                               to_torch([0, 1, 0], device=self.device).repeat(
                                                                   self.num_envs, 1) * -0.02)

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

        robot_state = self.compute_robot_state(full_obs=True)
        object_state = self.compute_object_state(set_goal=False)
        base_state = torch.cat((robot_state, object_state), dim=1)
        base_state = torch.clamp(base_state, -self.cfg["env"]["clip_observations"],
                                 self.cfg["env"]["clip_observations"])
        # base_state = robot_state

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


    def compute_object_state(self, set_goal=False):
        pass

    def compute_robot_state(self, full_obs=False):
        # dof_state = self.dof_state.view(self.num_envs, -1, 2)
        # dof_pos = dof_state[..., 0]
        # dof_vel = dof_state[..., 1]
        # robot_state = torch.cat((dof_pos, dof_vel), dim=1)

        robot_qpos = unscale(self.shadow_hand_dof_pos, self.shadow_hand_dof_lower_limits,
                             self.shadow_hand_dof_upper_limits)
        robot_qves = self.shadow_hand_dof_vel
        robot_state = torch.cat((robot_qpos, robot_qves), dim=1)

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

    def compute_pixel_obs(self, save_img=False):
        self.gym.render_all_camera_sensors(self.sim)
        self.gym.start_access_image_tensors(self.sim)
        self.im_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float, device=self.device).view(3, 1, 1)
        self.im_std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float, device=self.device).view(3, 1, 1)
        self.im_size = self.camera_props.width
        for i in range(self.num_envs):
            crop_l = (self.cam_w - self.im_size) // 2
            crop_r = crop_l + self.im_size
            self.img_buf[i] = self.camera_rgb_tensor_list[i][0][:, crop_l:crop_r, :3].float()  # / 255.
            if save_img:
                import cv2
                cv2.imwrite('2.png', np.array(self.img_buf[i].cpu()))

        self.gym.end_access_image_tensors(self.sim)
        pixel_obs = torch.flatten(self.img_buf, start_dim=1, end_dim=-1)
        return pixel_obs

    def compute_sensor_obs(self):
        # forces and torques
        contact = self.contact_force[self.hand_contact_idx].view(self.num_envs, self.num_force_sensors, 3)
        # vec_sensor = self.vec_sensor_tensor
        vec_sensor = contact
        vec_sensor = torch.norm(vec_sensor, p=2, dim=2)
        self.sensor_obs = torch.zeros_like(vec_sensor)
        self.sensor_obs[vec_sensor > 0.01] = 1
        # print(vec_sensor)
        return self.sensor_obs

    def reset_idx(self, env_ids, goal_env_ids):
        # randomization can happen only at reset time, since it can reset actor positions on GPU
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

        # generate random values
        rand_floats = torch_rand_float(-1.0, 1.0, (len(env_ids), self.num_shadow_hand_dofs * 2 + 5), device=self.device)

        # randomize start object poses
        self.reset_target_pose(env_ids)

        # reset rigid body forces
        self.rb_forces[env_ids, :, :] = 0.0

        # reset object
        self._reset_object(env_ids, rand_floats, self.object_indices[env_ids])
        object_indices = torch.unique(torch.cat([self.object_indices[env_ids],
                                                 self.goal_object_indices[env_ids],
                                                 self.goal_object_indices[goal_env_ids]]).to(torch.int32))
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_state_tensor),
                                                     gymtorch.unwrap_tensor(object_indices), len(object_indices))

        # reset random force probabilities
        self.random_force_prob[env_ids] = torch.exp((torch.log(self.force_prob_range[0]) - torch.log(self.force_prob_range[1]))
                                                    * torch.rand(len(env_ids), device=self.device) + torch.log(self.force_prob_range[1]))

        # reset shadow hand
        hand_indices = self.hand_indices[env_ids].to(torch.int32)
        self._reset_hand(env_ids, rand_floats, hand_indices)
        self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self.prev_targets),
                                                        gymtorch.unwrap_tensor(hand_indices), len(env_ids))

        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(hand_indices), len(env_ids))

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self.successes[env_ids] = 0

    def reset_target_pose(self, env_ids, apply_reset=False):
        rand_floats = torch_rand_float(-1.0, 1.0, (len(env_ids), 4), device=self.device)

        new_rot = randomize_rotation(rand_floats[:, 0], rand_floats[:, 1], self.x_unit_tensor[env_ids], self.y_unit_tensor[env_ids])

        self.goal_states[env_ids, 0:3] = self.goal_init_state[env_ids, 0:3]
        self.goal_states[env_ids, 3:7] = new_rot
        self.root_state_tensor[self.goal_object_indices[env_ids], 0:3] = self.goal_states[env_ids, 0:3] + self.goal_displacement_tensor
        self.root_state_tensor[self.goal_object_indices[env_ids], 3:7] = self.goal_states[env_ids, 3:7]
        self.root_state_tensor[self.goal_object_indices[env_ids], 7:13] = torch.zeros_like(self.root_state_tensor[self.goal_object_indices[env_ids], 7:13])

        if apply_reset:
            goal_object_indices = self.goal_object_indices[env_ids].to(torch.int32)
            self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                         gymtorch.unwrap_tensor(self.root_state_tensor),
                                                         gymtorch.unwrap_tensor(goal_object_indices), len(env_ids))
        self.reset_goal_buf[env_ids] = 0

    def _reset_hand(self, env_ids, rand_floats, hand_indices):

        delta_max = self.shadow_hand_dof_upper_limits - self.shadow_hand_dof_default_pos
        delta_min = self.shadow_hand_dof_lower_limits - self.shadow_hand_dof_default_pos
        # rand_delta = delta_min + (delta_max - delta_min) * 0.5 * (rand_floats[:, 5:5 + self.num_shadow_hand_dofs] + 1)

        dof_pos = self.shadow_hand_default_dof_pos  # + self.reset_dof_pos_noise * rand_delta
        self.shadow_hand_dof_pos[env_ids, :] = dof_pos
        self.shadow_hand_dof_vel[env_ids,
        :] = self.shadow_hand_dof_default_vel  # + self.reset_dof_vel_noise * rand_floats[:, 5 + self.num_shadow_hand_dofs:5 + self.num_shadow_hand_dofs * 2]
        self.prev_targets[env_ids, :self.num_shadow_hand_dofs] = dof_pos
        self.cur_targets[env_ids, :self.num_shadow_hand_dofs] = dof_pos

        self.hand_positions[hand_indices, :] = self.saved_root_tensor[hand_indices, 0:3]
        self.hand_orientations[hand_indices, :] = self.saved_root_tensor[hand_indices, 3:7]
        self.hand_linvels[hand_indices, :] = 0
        self.hand_angvels[hand_indices, :] = 0

        return

    def _reset_object(self, env_ids, rand_floats, object_indices):
        self.root_state_tensor[object_indices] = self.object_init_state[env_ids].clone()
        self.root_state_tensor[object_indices, 0:2] = self.object_init_state[env_ids, 0:2] + \
                                                                    self.reset_position_noise * rand_floats[:, 0:2]
        self.root_state_tensor[object_indices, self.up_axis_idx] = self.object_init_state[
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

        self.root_state_tensor[object_indices, 3:7] = new_object_rot
        self.root_state_tensor[object_indices, 7:13] = torch.zeros_like(
            self.root_state_tensor[object_indices, 7:13])
        return

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
            self.cur_targets[:, self.actuated_dof_indices] = scale(self.actions,
                                                                   self.shadow_hand_dof_lower_limits[self.actuated_dof_indices], self.shadow_hand_dof_upper_limits[self.actuated_dof_indices])
            self.cur_targets[:, self.actuated_dof_indices] = self.act_moving_average * self.cur_targets[:,
                                                                                                        self.actuated_dof_indices] + (1.0 - self.act_moving_average) * self.prev_targets[:, self.actuated_dof_indices]
            self.cur_targets[:, self.actuated_dof_indices] = tensor_clamp(self.cur_targets[:, self.actuated_dof_indices],
                                                                          self.shadow_hand_dof_lower_limits[self.actuated_dof_indices], self.shadow_hand_dof_upper_limits[self.actuated_dof_indices])

        self.prev_targets[:, self.actuated_dof_indices] = self.cur_targets[:, self.actuated_dof_indices]
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.cur_targets))

        # if self.force_scale > 0.0:
        #     self.rb_forces *= torch.pow(self.force_decay, self.dt / self.force_decay_interval)
        #
        #     # apply new forces
        #     force_indices = (torch.rand(self.num_envs, device=self.device) < self.random_force_prob).nonzero()
        #     self.rb_forces[force_indices, self.object_rb_handles, :] = torch.randn(
        #         self.rb_forces[force_indices, self.object_rb_handles, :].shape, device=self.device) * self.object_rb_masses * self.force_scale
        #
        #     self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(self.rb_forces), None, gymapi.LOCAL_SPACE)

    def post_physics_step(self):
        self.progress_buf += 1
        self.randomize_buf += 1

        self.compute_observations()
        self.compute_reward(self.actions)

        if self.viewer and self.debug_viz:
            # draw axes on target object
            self.gym.clear_lines(self.viewer)
            self.gym.refresh_rigid_body_state_tensor(self.sim)

            # for i in range(self.num_envs):
            #     targetx = (self.goal_pos[i] + quat_apply(self.goal_rot[i], to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
            #     targety = (self.goal_pos[i] + quat_apply(self.goal_rot[i], to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
            #     targetz = (self.goal_pos[i] + quat_apply(self.goal_rot[i], to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()
            #
            #     p0 = self.goal_pos[i].cpu().numpy() + self.goal_displacement_tensor.cpu().numpy()
            #     self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], targetx[0], targetx[1], targetx[2]], [0.85, 0.1, 0.1])
            #     self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], targety[0], targety[1], targety[2]], [0.1, 0.85, 0.1])
            #     self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], targetz[0], targetz[1], targetz[2]], [0.1, 0.1, 0.85])
            #
            #     objectx = (self.object_pos[i] + quat_apply(self.object_rot[i], to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
            #     objecty = (self.object_pos[i] + quat_apply(self.object_rot[i], to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
            #     objectz = (self.object_pos[i] + quat_apply(self.object_rot[i], to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()
            #
            #     p0 = self.object_pos[i].cpu().numpy()
            #     self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], objectx[0], objectx[1], objectx[2]], [0.85, 0.1, 0.1])
            #     self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], objecty[0], objecty[1], objecty[2]], [0.1, 0.85, 0.1])
            #     self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], objectz[0], objectz[1], objectz[2]], [0.1, 0.1, 0.85])

#####################################################################
###=========================jit functions=========================###
#####################################################################


@torch.jit.script
def compute_hand_reward(
    rew_buf, reset_buf, reset_goal_buf, progress_buf, successes, consecutive_successes,
    max_episode_length: float, object_pos, object_rot, target_pos, target_rot,
    dist_reward_scale: float, rot_reward_scale: float, rot_eps: float,
    actions, action_penalty_scale: float,
    success_tolerance: float, reach_goal_bonus: float, fall_dist: float,
    fall_penalty: float, max_consecutive_successes: int, av_factor: float, ignore_z_rot: bool
):
    # Distance from the hand to the object
    goal_dist = torch.norm(object_pos - target_pos, p=2, dim=-1)

    if ignore_z_rot:
        success_tolerance = 2.0 * success_tolerance

    # Orientation alignment for the cube in hand and goal cube
    quat_diff = quat_mul(object_rot, quat_conjugate(target_rot))
    rot_dist = 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 0:3], p=2, dim=-1), max=1.0))

    dist_rew = goal_dist * dist_reward_scale
    rot_rew = 1.0/(torch.abs(rot_dist) + rot_eps) * rot_reward_scale

    action_penalty = torch.sum(actions ** 2, dim=-1)

    # Total reward is: position distance + orientation alignment + action regularization + success bonus + fall penalty
    reward = dist_rew + rot_rew + action_penalty * action_penalty_scale

    # Find out which envs hit the goal and update successes count
    goal_resets = torch.where(torch.abs(rot_dist) <= success_tolerance, torch.ones_like(reset_goal_buf), reset_goal_buf)
    successes = successes + goal_resets

    # Success bonus: orientation is within `success_tolerance` of goal orientation
    reward = torch.where(goal_resets == 1, reward + reach_goal_bonus, reward)

    # Fall penalty: distance to the goal is larger than a threshold
    reward = torch.where(goal_dist >= fall_dist, reward + fall_penalty, reward)

    # Check env termination conditions, including maximum success number
    resets = torch.where(goal_dist >= fall_dist, torch.ones_like(reset_buf), reset_buf)
    if max_consecutive_successes > 0:
        # Reset progress buffer on goal envs if max_consecutive_successes > 0
        progress_buf = torch.where(torch.abs(rot_dist) <= success_tolerance, torch.zeros_like(progress_buf), progress_buf)
        resets = torch.where(successes >= max_consecutive_successes, torch.ones_like(resets), resets)
    resets = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(resets), resets)

    # Apply penalty for not reaching the goal
    if max_consecutive_successes > 0:
        reward = torch.where(progress_buf >= max_episode_length - 1, reward + 0.5 * fall_penalty, reward)

    num_resets = torch.sum(resets)
    finished_cons_successes = torch.sum(successes * resets.float())

    cons_successes = torch.where(num_resets > 0, av_factor*finished_cons_successes/num_resets + (1.0 - av_factor)*consecutive_successes, consecutive_successes)

    return reward, resets, goal_resets, progress_buf, successes, cons_successes


@torch.jit.script
def randomize_rotation(rand0, rand1, x_unit_tensor, y_unit_tensor):
    return quat_mul(quat_from_angle_axis(rand0 * np.pi, x_unit_tensor),
                    quat_from_angle_axis(rand1 * np.pi, y_unit_tensor))


@torch.jit.script
def randomize_rotation_pen(rand0, rand1, max_angle, x_unit_tensor, y_unit_tensor, z_unit_tensor):
    rot = quat_mul(quat_from_angle_axis(0.5 * np.pi + rand0 * max_angle, x_unit_tensor),
                   quat_from_angle_axis(rand0 * np.pi, z_unit_tensor))
    return rot

# @torch.jit.script
# def mov(tensor, device):
#     return torch.from_numpy(tensor.cpu().numpy()).to(device)