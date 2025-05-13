#!/usr/bin/env python3

"""Utils."""
import json

import hydra
import numpy as np
import os
import random

from omegaconf import DictConfig, OmegaConf
from typing import Dict

from isaacgym import gymapi
from isaacgym import gymutil

import torch

from tv_tasks.tasks.base.vec_task import VecTaskPython
from tv_tasks.tasks import *

# Available tasks
_TASK_MAP = {
    "BottleCap": BottleCap,
    "ScrewFaucet": ScrewFaucet,
    "ReorientUp": ReorientUp,
    "Sliding": Sliding,
    "ReorientDown": ReorientDown,
    "HandOver": HandOver,
}


def set_np_formatting():
    np.set_printoptions(edgeitems=30, infstr='inf',
                        linewidth=4000, nanstr='nan', precision=2,
                        suppress=False, threshold=10000, formatter=None)


def set_seed(seed, torch_deterministic=False):
    if seed == -1 and torch_deterministic:
        seed = 42
    elif seed == -1:
        seed = np.random.randint(0, 10000)
    print("Setting seed: {}".format(seed))

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if torch_deterministic:
        # refer to https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.set_deterministic(True)
        # torch.use_deterministic_algorithms(True)

    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    return seed


def parse_sim_params(args):
    # previously args defaults
    args_use_gpu_pipeline = (args.pipeline in ["gpu", "cuda"])
    args_use_gpu = ("cuda" in args.sim_device)
    args_subscenes = 0
    args_slices = args_subscenes
    args_num_threads = 0

    # initialize sim
    sim_params = gymapi.SimParams()
    sim_params.dt = 1 / 60.
    sim_params.num_client_threads = args_slices

    assert args.physics_engine == "physx"
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 4
    sim_params.physx.num_velocity_iterations = 0
    sim_params.physx.num_threads = 4
    sim_params.physx.use_gpu = args_use_gpu
    sim_params.physx.num_subscenes = args_subscenes
    sim_params.physx.max_gpu_contact_pairs = 8 * 1024 * 1024

    sim_params.use_gpu_pipeline = args_use_gpu_pipeline
    sim_params.physx.use_gpu = args_use_gpu

    # if sim options are provided in cfg parse them and update/override above:
    if "sim" in args.task_envs:
        print("Setting sim options")
        gymutil.parse_sim_config(args.task_envs["sim"], sim_params)

    # Override num_threads if specified
    if args.physics_engine == "physx" and args_num_threads > 0:
        sim_params.physx.num_threads = args_num_threads

    return sim_params


def parse_task(args, sim_params):
    physics_engine = gymapi.SIM_PHYSX if args.physics_engine == "physx" else gymapi.SIM_GLEX
    device_type = "cuda" if "cuda" in args.sim_device else "cpu"
    device_id = int(args.graphics_device_id.split(":")[1]) if "cuda" in args.graphics_device_id else 0
    headless = args.headless
    rl_device = args.rl_device

    if args.num_gpus > 1:
        curr_device = torch.cuda.current_device()
        device_id = curr_device
        rl_device = curr_device

    task = _TASK_MAP[args.task_envs['task_name']](
        cfg=args.task_envs,
        sim_params=sim_params,
        physics_engine=physics_engine,
        device_type=device_type,
        device_id=device_id,
        headless=headless
    )

    env = VecTaskPython(task, rl_device)

    return env


def warn_task_name():
    raise Exception(
        "Unrecognized task!")


def warn_algorithm_name():
    raise Exception(
        "Unrecognized algorithm!\nAlgorithm should be one of: [ppo, happo, hatrpo, mappo]")


import os
import argparse
import yaml
from utils.util import pretty
from utils.util import DotDict
import torch


def get_args():
    parser = argparse.ArgumentParser(description='manipulation tasks')

    parser.add_argument('--task', type=str, required=True, help='manipulated task for an agent')
    parser.add_argument('--algo', type=str, default='ppo', help='training algorithm')
    parser.add_argument('--resume_model', type=str, default='', help='Choose a model dir')
    parser.add_argument('--sim_device', type=str, default='cuda:0', help='device for simulation')
    parser.add_argument('--rl_device', type=str, default='cuda:0', help='device for training')
    parser.add_argument('--headless', action='store_true', help='visualize envs or not')
    parser.add_argument('--test', action='store_true', help='test')
    parser.add_argument('--seed', required=True, type=int, help='test')
    parser.add_argument('--physics_engine', type=str, default='physx', help='Choose a physics engine: physx or flex')
    parser.add_argument('--pipeline', type=str, default='gpu', help='sim data on gpu or cpu')
    parser.add_argument('--graphics_device_id', type=int, default='0', help='graphics_device_id')
    parser.add_argument('--num_gpus', type=int, default='1', help='')
    parser.add_argument('--identifier', type=str, default='model_single3', help='')
    parser.add_argument('--all_train', action='store_true', help='train all objects or not')

    parser.add_argument('--allegro', action='store_true', help='allegro hand')

    args = parser.parse_args()
    # args.sim_device = args.rl_device # move to line 237
    args.graphics_device_id = args.rl_device

    if args.allegro:
        args.task = args.task + '_allegro_hand'

    assert args.task.split("-")[0] + '.yaml' in os.listdir(
        './config/task_env'), f"task is not defined, please choose one from {os.listdir('./config/task_env')}(except '.yaml')"
    with open('./config/task_env/' + args.task.split("-")[0] + '.yaml') as f:
        args.task_envs = yaml.load(f, Loader=yaml.FullLoader)

    with open(f'./config/algos/{args.algo}/{args.task.split("-")[0]}.yaml') as f:
        args.models = yaml.load(f, Loader=yaml.FullLoader)
    args.models['seed'] = args.seed

    if args.task.split("-")[0] == "bottle_cap":
        args.identifier = "bottle_cap" #dp05, dp05_rn, dp05_rn2
        args.sim_device = "cpu"
        args.pipeline = "cpu"
    elif args.task.split("-")[0] == "screw_faucet":
        args.identifier = "screw_faucet" #dp05, dp05_rn, dp05_rn2
        args.sim_device = "cpu"
        args.pipeline = "cpu"
    elif args.task.split("-")[0] == "reorient_down":
        args.identifier = "reorient_down"
        args.sim_device = "cpu"
        args.pipeline = "cpu"
    elif args.task.split("-")[0] == "reorient_up":
        args.identifier = "reorient_up"
        args.sim_device = "cpu"
        args.pipeline = "cpu"
    elif args.task.split("-")[0] == "handover":
        args.identifier = "handover"
        # args.sim_device = "cpu"
        # args.pipeline = "cpu"
    elif args.task.split("-")[0] == "slide":
        args.identifier = "slide"
        args.sim_device = "cpu"
        args.pipeline = "cpu"
    else:
        args.sim_device = args.rl_device
        # args.sim_device = "cpu"
        # args.pipeline = "cpu"
    print(f"Task: {args.task.split('-')[0]}")
    print(f"Identifier: {args.identifier}")

    if args.task.split("-")[-1] == "vt_all_cls":
        args.models["encoder"]["name"] = "vt20t-reall-tmr05-bin-ft-cls+dataset-ViTacReal-all-210"
        args.models["policy"]["actor_critic"] = "ActorCriticVTEncoder"
        args.task_envs["env"]["obs_type"] = "VisTac"
        args.task_envs["env"]["obs_dim"]["vis"] = 150528
        args.task_envs["env"]["obs_dim"]["tac"] = 20
        args.models["encoder"]["en_mode"] = "cls"
    elif args.task.split("-")[-1] == "vt_all_cls_MV":
        args.models["encoder"]["name"] = "vt20t-reall-tmr05-bin-ft-cls+dataset-ViTacReal-all-210"
        args.models["policy"]["actor_critic"] = "ActorCriticVTEncoder"
        args.task_envs["env"]["obs_type"] = "VisTac"
        args.task_envs["env"]["obs_dim"]["vis"] = 150528
        args.task_envs["env"]["obs_dim"]["tac"] = 20
        args.models["encoder"]["en_mode"] = "cls"
        args.task_envs["env"]["masked_modality"] = "vis"
        print(f"masked_modality: {args.task_envs['env']['masked_modality']}")
    elif args.task.split("-")[-1] == "vt_all_cls_MT":
        args.models["encoder"]["name"] = "vt20t-reall-tmr05-bin-ft-cls+dataset-ViTacReal-all-210"
        args.models["policy"]["actor_critic"] = "ActorCriticVTEncoder"
        args.task_envs["env"]["obs_type"] = "VisTac"
        args.task_envs["env"]["obs_dim"]["vis"] = 150528
        args.task_envs["env"]["obs_dim"]["tac"] = 20
        args.models["encoder"]["en_mode"] = "cls"
        args.task_envs["env"]["masked_modality"] = "tac"
        print(f"masked_modality: {args.task_envs['env']['masked_modality']}")
    elif args.task.split("-")[-1] == "vt_all_cls_tacts05":
        args.models["encoder"]["name"] = "vt20t-reall-tmr05-bin-ft-cls+dataset-ViTacReal-all-210"
        args.models["policy"]["actor_critic"] = "ActorCriticVTEncoder"
        args.task_envs["env"]["obs_type"] = "VisTac"
        args.task_envs["env"]["obs_dim"]["vis"] = 150528
        args.task_envs["env"]["obs_dim"]["tac"] = 20
        args.models["encoder"]["en_mode"] = "cls"
        args.task_envs["env"]["tac_theshold"] = 0.5
    elif args.task.split("-")[-1] == "vt_all_cls_tacts1":
        args.models["encoder"]["name"] = "vt20t-reall-tmr05-bin-ft-cls+dataset-ViTacReal-all-210"
        args.models["policy"]["actor_critic"] = "ActorCriticVTEncoder"
        args.task_envs["env"]["obs_type"] = "VisTac"
        args.task_envs["env"]["obs_dim"]["vis"] = 150528
        args.task_envs["env"]["obs_dim"]["tac"] = 20
        args.models["encoder"]["en_mode"] = "cls"
        args.task_envs["env"]["tac_theshold"] = 1.0

    elif args.task.split("-")[-1] == "vt_all_cls_tn001":
        args.models["encoder"]["name"] = "vt20t-reall-tmr05-bin-ft-cls+dataset-ViTacReal-all-210"
        args.models["policy"]["actor_critic"] = "ActorCriticVTEncoder"
        args.task_envs["env"]["obs_type"] = "VisTac"
        args.task_envs["env"]["obs_dim"]["vis"] = 150528
        args.task_envs["env"]["obs_dim"]["tac"] = 20
        args.models["encoder"]["en_mode"] = "cls"
        args.task_envs["env"]["tac_noise"] = 0.01
    elif args.task.split("-")[-1] == "vt_all_cls_tn01":
        args.models["encoder"]["name"] = "vt20t-reall-tmr05-bin-ft-cls+dataset-ViTacReal-all-210"
        args.models["policy"]["actor_critic"] = "ActorCriticVTEncoder"
        args.task_envs["env"]["obs_type"] = "VisTac"
        args.task_envs["env"]["obs_dim"]["vis"] = 150528
        args.task_envs["env"]["obs_dim"]["tac"] = 20
        args.models["encoder"]["en_mode"] = "cls"
        args.task_envs["env"]["tac_noise"] = 0.1
    elif args.task.split("-")[-1] == "vt_all_cls_tn1":
        args.models["encoder"]["name"] = "vt20t-reall-tmr05-bin-ft-cls+dataset-ViTacReal-all-210"
        args.models["policy"]["actor_critic"] = "ActorCriticVTEncoder"
        args.task_envs["env"]["obs_type"] = "VisTac"
        args.task_envs["env"]["obs_dim"]["vis"] = 150528
        args.task_envs["env"]["obs_dim"]["tac"] = 20
        args.models["encoder"]["en_mode"] = "cls"
        args.task_envs["env"]["tac_noise"] = 1.0
    elif args.task.split("-")[-1] == "vt_all_cls_sep":
        args.models["encoder"]["name"] = ["v-repic-bin-ft-cls+dataset-ViTacReal-all-170",
                                          "t20-retac-tmr05-bin-ft-cls+dataset-ViTacReal-all-310"]
        args.models["policy"]["actor_critic"] = "ActorCriticV_TEncoder"
        args.task_envs["env"]["obs_type"] = "VisTac"
        args.task_envs["env"]["obs_dim"]["vis"] = 150528
        args.task_envs["env"]["obs_dim"]["tac"] = 20
        args.models["encoder"]["en_mode"] = "cls"
    elif args.task.split("-")[-1] == "vt_all_cls_fullocc":
        args.models["encoder"]["name"] = "vt20t-reall-tmr05-bin-ft-cls+dataset-ViTacReal-all-210"
        args.models["policy"]["actor_critic"] = "ActorCriticVTEncoder"
        args.task_envs["env"]["obs_type"] = "VisTac"
        args.task_envs["env"]["obs_dim"]["vis"] = 150528
        args.task_envs["env"]["obs_dim"]["tac"] = 20
        args.models["encoder"]["en_mode"] = "cls"
        print("vt_all_cls_fullocc")
        if args.task.split("-")[0] == "bottle_cap":
            args.task_envs["env"]["vision"]["camera"]["eye"] = [[-0.25, 0, 0.5]]
            args.task_envs["env"]["vision"]["camera"]["lookat"] = [[0.02, 0, 0.1]]
        elif args.task.split("-")[0] == "screw_faucet":
            args.task_envs["env"]["vision"]["camera"]["eye"] = [[0.4, 0, 0.6]]
            args.task_envs["env"]["vision"]["camera"]["lookat"] = [[0, 0, 0.05]]
        elif args.task.split("-")[0] == "reorient_down":
            args.task_envs["env"]["vision"]["camera"]["eye"] = [[ 0, 0.3, 0.4 ]]
            args.task_envs["env"]["vision"]["camera"]["lookat"] = [[0, 0, 0.05]]
        else:
            raise AttributeError

    elif args.task.split("-")[-1] == "vt_all_cls_fullvis":
        args.models["encoder"]["name"] = "vt20t-reall-tmr05-bin-ft-cls+dataset-ViTacReal-all-210"
        args.models["policy"]["actor_critic"] = "ActorCriticVTEncoder"
        args.task_envs["env"]["obs_type"] = "VisTac"
        args.task_envs["env"]["obs_dim"]["vis"] = 150528
        args.task_envs["env"]["obs_dim"]["tac"] = 20
        args.models["encoder"]["en_mode"] = "cls"
        print("vt_all_cls_fullvis")
        if args.task.split("-")[0] == "bottle_cap":
            args.task_envs["env"]["vision"]["camera"]["eye"] = [[0.25, 0.35, 0.5]]
            args.task_envs["env"]["vision"]["camera"]["lookat"] = [[-0.02, -0.02, 0.1]]
        elif args.task.split("-")[0] == "screw_faucet":
            args.task_envs["env"]["vision"]["camera"]["eye"] = [[-0.4, -0.4, 0.4]]
            args.task_envs["env"]["vision"]["camera"]["lookat"] = [[0, 0.05, 0.05]]
        elif args.task.split("-")[0] == "reorient_down":
            args.task_envs["env"]["vision"]["camera"]["eye"] = [[ 0.2, -0.2, 0.3 ]]
            args.task_envs["env"]["vision"]["camera"]["lookat"] = [[0, 0, 0.05]]
        else:
            raise AttributeError
    elif args.task.split("-")[-1] == "vis_all_cls_fullocc":
        args.models["encoder"]["name"] = "v-repic-bin-ft-cls+dataset-ViTacReal-all-170"
        args.models["policy"]["actor_critic"] = "ActorCriticVEncoder"
        args.task_envs["env"]["obs_type"] = "VisOnly"
        args.task_envs["env"]["obs_dim"]["vis"] = 150528
        args.models["encoder"]["en_mode"] = "cls"
        print("vis_all_cls_fullocc")
        if args.task.split("-")[0] == "bottle_cap":
            args.task_envs["env"]["vision"]["camera"]["eye"] = [[-0.25, 0, 0.5]]
            args.task_envs["env"]["vision"]["camera"]["lookat"] = [[0.02, 0, 0.1]]
        elif args.task.split("-")[0] == "screw_faucet":
            args.task_envs["env"]["vision"]["camera"]["eye"] = [[0.4, 0, 0.6]]
            args.task_envs["env"]["vision"]["camera"]["lookat"] = [[0, 0, 0.05]]
        elif args.task.split("-")[0] == "reorient_down":
            args.task_envs["env"]["vision"]["camera"]["eye"] = [[ 0, 0.3, 0.4 ]]
            args.task_envs["env"]["vision"]["camera"]["lookat"] = [[0, 0, 0.05]]
        else:
            raise AttributeError

    elif args.task.split("-")[-1] == "vis_all_cls_fullvis":
        args.models["encoder"]["name"] = "v-repic-bin-ft-cls+dataset-ViTacReal-all-170"
        args.models["policy"]["actor_critic"] = "ActorCriticVEncoder"
        args.task_envs["env"]["obs_type"] = "VisOnly"
        args.task_envs["env"]["obs_dim"]["vis"] = 150528
        args.models["encoder"]["en_mode"] = "cls"
        print("vis_all_cls_fullvis")
        if args.task.split("-")[0] == "bottle_cap":
            args.task_envs["env"]["vision"]["camera"]["eye"] = [[0.25, 0.35, 0.5]]
            args.task_envs["env"]["vision"]["camera"]["lookat"] = [[-0.02, -0.02, 0.1]]
        elif args.task.split("-")[0] == "screw_faucet":
            args.task_envs["env"]["vision"]["camera"]["eye"] = [[-0.4, -0.4, 0.4]]
            args.task_envs["env"]["vision"]["camera"]["lookat"] = [[0, 0.05, 0.05]]
        elif args.task.split("-")[0] == "reorient_down":
            args.task_envs["env"]["vision"]["camera"]["eye"] = [[ 0.2, -0.2, 0.3 ]]
            args.task_envs["env"]["vision"]["camera"]["lookat"] = [[0, 0, 0.05]]
        else:
            raise AttributeError
    elif args.task.split("-")[-1] == "vis_all_cls":
        args.models["encoder"]["name"] = "v-repic-bin-ft-cls+dataset-ViTacReal-all-170"
        args.models["policy"]["actor_critic"] = "ActorCriticVEncoder"
        args.task_envs["env"]["obs_type"] = "VisOnly"
        args.task_envs["env"]["obs_dim"]["vis"] = 150528
        args.models["encoder"]["en_mode"] = "cls"
    elif args.task.split("-")[-1] == "tac_all_cls":
        args.models["encoder"]["name"] = "t20-retac-tmr05-bin-ft-cls+dataset-ViTacReal-all-310"
        args.models["policy"]["actor_critic"] = "ActorCriticTEncoder"
        args.task_envs["env"]["obs_type"] = "TacOnly"
        args.task_envs["env"]["obs_dim"]["tac"] = 20
        args.models["encoder"]["en_mode"] = "cls"
    elif args.task.split("-")[-1] == "tac_all_cls_tacts1":
        args.models["encoder"]["name"] = "t20-retac-tmr05-bin-ft-cls+dataset-ViTacReal-all-310"
        args.models["policy"]["actor_critic"] = "ActorCriticTEncoder"
        args.task_envs["env"]["obs_type"] = "TacOnly"
        args.task_envs["env"]["obs_dim"]["tac"] = 20
        args.models["encoder"]["en_mode"] = "cls"
        args.task_envs["env"]["tac_theshold"] = 1.0
    elif args.task.split("-")[-1] == "tac_all_cls_tacts05":
        args.models["encoder"]["name"] = "t20-retac-tmr05-bin-ft-cls+dataset-ViTacReal-all-310"
        args.models["policy"]["actor_critic"] = "ActorCriticTEncoder"
        args.task_envs["env"]["obs_type"] = "TacOnly"
        args.task_envs["env"]["obs_dim"]["tac"] = 20
        args.models["encoder"]["en_mode"] = "cls"
        args.task_envs["env"]["tac_theshold"] = 0.5

    elif args.task.split("-")[-1] == "tac_all_cls_tn001":
        args.models["encoder"]["name"] = "t20-retac-tmr05-bin-ft-cls+dataset-ViTacReal-all-310"
        args.models["policy"]["actor_critic"] = "ActorCriticTEncoder"
        args.task_envs["env"]["obs_type"] = "TacOnly"
        args.task_envs["env"]["obs_dim"]["tac"] = 20
        args.models["encoder"]["en_mode"] = "cls"
        args.task_envs["env"]["tac_noise"] = 0.01
    elif args.task.split("-")[-1] == "tac_all_cls_tn01":
        args.models["encoder"]["name"] = "t20-retac-tmr05-bin-ft-cls+dataset-ViTacReal-all-310"
        args.models["policy"]["actor_critic"] = "ActorCriticTEncoder"
        args.task_envs["env"]["obs_type"] = "TacOnly"
        args.task_envs["env"]["obs_dim"]["tac"] = 20
        args.models["encoder"]["en_mode"] = "cls"
        args.task_envs["env"]["tac_noise"] = 0.1
    elif args.task.split("-")[-1] == "tac_all_cls_tn1":
        args.models["encoder"]["name"] = "t20-retac-tmr05-bin-ft-cls+dataset-ViTacReal-all-310"
        args.models["policy"]["actor_critic"] = "ActorCriticTEncoder"
        args.task_envs["env"]["obs_type"] = "TacOnly"
        args.task_envs["env"]["obs_dim"]["tac"] = 20
        args.models["encoder"]["en_mode"] = "cls"
        args.task_envs["env"]["tac_noise"] = 1.0

    elif args.task.split("-")[-1] == "vt_resnet18":
        args.models["encoder"]["name"] = "resnet18"
        args.models["policy"]["actor_critic"] = "ActorCriticVT"
        args.models["learn"]["nminibatches"] = 16
        args.task_envs["env"]["obs_type"] = "VisTac"
        args.task_envs["env"]["obs_dim"]["vis"] = 150528
        args.task_envs["env"]["obs_dim"]["tac"] = 20
        args.task_envs["env"]["numEnvs"] = 200
    elif args.task.split("-")[-1] == "v_resnet18":
        args.models["encoder"]["name"] = "resnet18"
        args.models["policy"]["actor_critic"] = "ActorCriticV"
        args.models["learn"]["nminibatches"] = 16
        args.task_envs["env"]["obs_type"] = "VisOnly"
        args.task_envs["env"]["obs_dim"]["vis"] = 150528
        # args.task_envs["env"]["obs_dim"]["tac"] = 20
        # args.task_envs["env"]["numEnvs"] = 200
    elif args.task.split("-")[-1] == "t_scr":
        args.models["encoder"]["name"] = "mlp"
        args.models["policy"]["actor_critic"] = "ActorCriticT"
        args.models["learn"]["nminibatches"] = 4
        args.task_envs["env"]["obs_type"] = "TacOnly"
        # args.task_envs["env"]["obs_dim"]["vis"] = 150528
        args.task_envs["env"]["obs_dim"]["tac"] = 20
    elif args.task.split("-")[-1] == "t_scr_tn01":
        args.models["encoder"]["name"] = "mlp"
        args.models["policy"]["actor_critic"] = "ActorCriticT"
        args.models["learn"]["nminibatches"] = 4
        args.task_envs["env"]["obs_type"] = "TacOnly"
        args.task_envs["env"]["tac_noise"] = 0.1
        args.task_envs["env"]["obs_dim"]["tac"] = 20
    elif args.task.split("-")[-1] == "t_scr_tn001":
        args.models["encoder"]["name"] = "mlp"
        args.models["policy"]["actor_critic"] = "ActorCriticT"
        args.models["learn"]["nminibatches"] = 4
        args.task_envs["env"]["obs_type"] = "TacOnly"
        args.task_envs["env"]["obs_dim"]["tac"] = 20
        args.task_envs["env"]["tac_noise"] = 0.01
    elif args.task.split("-")[-1] == "t_scr_tn1":
        args.models["encoder"]["name"] = "mlp"
        args.models["policy"]["actor_critic"] = "ActorCriticT"
        args.models["learn"]["nminibatches"] = 4
        args.task_envs["env"]["obs_type"] = "TacOnly"
        args.task_envs["env"]["obs_dim"]["tac"] = 20
        args.task_envs["env"]["tac_noise"] = 1.0

    elif args.task.split("-")[-1] == "base":
        args.models["encoder"]["name"] = "vt20t-reall-tmr05-bin-ft+dataset-ViTacReal-900f"
        args.models["policy"]["actor_critic"] = "ActorCritic"
        args.task_envs["env"]["obs_type"] = "Base"

    elif args.task.split("-")[-1] == "mvp":
        args.models["encoder"]["name"] = "VMVP"
        args.models["encoder"]["en_mode"] = "cls"
        args.models["encoder"]["pretrain_dir"] = "model/reproductions/model_and_config"
        args.models["policy"]["actor_critic"] = "ActorCriticVEncoder"
        args.task_envs["env"]["obs_type"] = "VisOnly"
        args.task_envs["env"]["obs_dim"]["vis"] = 150528
    elif args.task.split("-")[-1] == "vt_resnet18_pre":
        args.models["encoder"]["name"] = "resnet18"
        args.models["encoder"]["en_mode"] = "cls"
        args.models["policy"]["actor_critic"] = "ActorCriticVEncoderT"
        args.task_envs["env"]["obs_type"] = "VisTac"
        args.task_envs["env"]["obs_dim"]["vis"] = 150528
        args.task_envs["env"]["obs_dim"]["tac"] = 20
        args.task_envs["env"]["numEnvs"] = 200
    elif args.task.split("-")[-1] == "v_resnet18_pre":
        args.models["encoder"]["name"] = "resnet18"
        args.models["encoder"]["en_mode"] = "cls"
        args.models["policy"]["actor_critic"] = "ActorCriticVEncoder"
        args.task_envs["env"]["obs_type"] = "VisOnly"
        args.task_envs["env"]["obs_dim"]["vis"] = 150528
    elif args.task.split("-")[-1] == "vt_clip":
        args.models["encoder"]["name"] = "CLIP"
        args.models["encoder"]["en_mode"] = "cls"
        args.models["policy"]["actor_critic"] = "ActorCriticVEncoderT"
        args.task_envs["env"]["obs_type"] = "VisTac"
        args.task_envs["env"]["obs_dim"]["vis"] = 150528
        args.task_envs["env"]["obs_dim"]["tac"] = 20
    elif args.task.split("-")[-1] == "vt_voltron":
        args.models["encoder"]["name"] = "Voltron"
        args.models["encoder"]["en_mode"] = "patch"
        args.models["encoder"]["f_ex_mode"] = "MAP"
        args.models["policy"]["actor_critic"] = "ActorCriticVEncoderT"
        args.task_envs["env"]["obs_type"] = "VisTac"
        args.task_envs["env"]["obs_dim"]["vis"] = 150528
        args.task_envs["env"]["obs_dim"]["tac"] = 20
    elif args.task.split("-")[-1] == "vt_r3m":
        args.models["encoder"]["name"] = "R3M"
        args.models["encoder"]["en_mode"] = "cls"
        args.models["policy"]["actor_critic"] = "ActorCriticVEncoderT"
        args.task_envs["env"]["obs_type"] = "VisTac"
        args.task_envs["env"]["obs_dim"]["vis"] = 150528
        args.task_envs["env"]["obs_dim"]["tac"] = 20
    elif args.task.split("-")[-1] == "vt_mvp":
        args.models["encoder"]["name"] = "MVP"
        args.models["encoder"]["en_mode"] = "cls"
        args.models["policy"]["actor_critic"] = "ActorCriticVEncoderT"
        args.task_envs["env"]["obs_type"] = "VisTac"
        args.task_envs["env"]["obs_dim"]["vis"] = 150528
        args.task_envs["env"]["obs_dim"]["tac"] = 20

    elif args.task.split("-")[-1] == "v_clip":
        args.models["encoder"]["name"] = "CLIP"
        args.models["encoder"]["en_mode"] = "cls"
        args.models["policy"]["actor_critic"] = "ActorCriticVEncoder"
        args.task_envs["env"]["obs_type"] = "VisOnly"
        args.task_envs["env"]["obs_dim"]["vis"] = 150528
    elif args.task.split("-")[-1] == "v_voltron":
        args.models["encoder"]["name"] = "Voltron"
        args.models["encoder"]["en_mode"] = "patch"
        args.models["encoder"]["f_ex_mode"] = "MAP"
        args.models["policy"]["actor_critic"] = "ActorCriticVEncoder"
        args.task_envs["env"]["obs_type"] = "VisOnly"
        args.task_envs["env"]["obs_dim"]["vis"] = 150528
    elif args.task.split("-")[-1] == "v_r3m":
        args.models["encoder"]["name"] = "R3M"
        args.models["encoder"]["en_mode"] = "cls"
        args.models["policy"]["actor_critic"] = "ActorCriticVEncoder"
        args.task_envs["env"]["obs_type"] = "VisOnly"
        args.task_envs["env"]["obs_dim"]["vis"] = 150528
    elif args.task.split("-")[-1] == "v_mvp":
        args.models["encoder"]["name"] = "MVP"
        args.models["encoder"]["en_mode"] = "cls"
        args.models["policy"]["actor_critic"] = "ActorCriticVEncoder"
        args.task_envs["env"]["obs_type"] = "VisOnly"
        args.task_envs["env"]["obs_dim"]["vis"] = 150528
        # args.task_envs["env"]["obs_dim"]["tac"] = 20
    else:
        raise AttributeError(f"{args.task.split('-')[-1]} not define")


    if args.test:
        args.models['learn']['test'] = args.test
        args.task_envs["env"]["numEnvs"] = len(args.task_envs["env"]["env_dict"])

    curr_dir = os.getcwd()
    args.logger_dir = os.path.join(curr_dir, 'runs', args.task_envs['task_name'], args.identifier, args.task,
                                   'seed' + str(args.models['seed']))
    os.makedirs(args.logger_dir, exist_ok=True)

    if args.all_train:
        with open(args.task_envs['env']['all_env_path'], 'r') as f:
            all_env_names = json.load(f)
        args.task_envs['env']['env_dict'] = all_env_names['train']
        args.task_envs['env']['numEnvs'] = len(args.task_envs['env']['env_dict']) * args.task_envs['env'][
            'n_env_repeat']
        args.task_envs['env']['env_test_dict'] = all_env_names['test']
    print(f"Num Envs --> {args.task_envs['env']['numEnvs']}")

    # args.train_params = args.task_envs['train_params']
    # del args.task_envs['train_params']
    if not args.models['learn']['test']:
        config_params = vars(args)
        # save all config
        with open(os.path.join(args.logger_dir, 'all_config.yaml'), 'w') as f:
            f.write(yaml.dump(config_params))

        # print(pretty(config_params))

    return args
