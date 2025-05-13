from utils.hydra_utils import parse_sim_params, parse_task, set_np_formatting, set_seed, get_args
from model.process_sarl import process_sarl

import torch
torch.backends.cudnn.enabled = False

def main():
    # Set up python env
    set_np_formatting()
    args = get_args()
    set_seed(args.models['seed'], args.models['torch_deterministic'])

    # Construct task
    sim_params = parse_sim_params(args)
    env = parse_task(args, sim_params)
    # set up policy

    sarl = process_sarl(args, env, args.models, args.logger_dir)
    sarl.run(num_learning_iterations=args.models["learn"]["max_iterations"], log_interval=args.models["learn"]["save_interval"])


if __name__ == '__main__':
    main()