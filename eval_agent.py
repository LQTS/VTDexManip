import warnings

from tqdm import tqdm

from utils.logger import DataLog
import os
from utils.hydra_utils import parse_sim_params, parse_task, set_np_formatting, set_seed, get_args
from model.process_sarl import process_sarl
import torch

# torch.backends.cudnn.benchmark = True


def eval_policy(max_trajs=100):

    # Set up python env
    set_np_formatting()
    args = get_args()
    set_seed(args.models['seed'], args.models['torch_deterministic'])

    # Construct task
    sim_params = parse_sim_params(args)
    env = parse_task(args, sim_params)
    # args.resume_model
    logger = DataLog()
    assert os.path.isdir(args.logger_dir)

    # model_path = os.path.join(args.logger_dir, 'checkpoint')

    while True:
        # args.resume_model = os.path.join(model_path, f'model_{2000}.pt')
        # set up policy
        sarl = process_sarl(args, env, args.models, args.logger_dir)
        sarl.eval(logger, max_trajs=max_trajs, record_video = False)



if __name__ == '__main__':
    eval_policy()