import pickle
from datetime import datetime
import os
import time
import math
import numpy as np
import statistics
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision.io import write_video
from model.ppo import RolloutStorage
from model.ppo.policy import ActorCritic, ActorCriticVEncoder, ActorCriticVTEncoder, ActorCriticVT, ActorCriticTEncoder, ActorCriticV_TEncoder, ActorCriticVEncoderT, ActorCriticV, ActorCriticT

import copy
from utils.tensorboard_extract import tensorboard2csv

_MODEL_FUNCS = {
    "ActorCritic": ActorCritic,
    "ActorCriticVEncoder": ActorCriticVEncoder,
    "ActorCriticVTEncoder": ActorCriticVTEncoder,
    "ActorCriticVT": ActorCriticVT,
    "ActorCriticTEncoder": ActorCriticTEncoder,
    "ActorCriticV_TEncoder": ActorCriticV_TEncoder,
    "ActorCriticVEncoderT": ActorCriticVEncoderT,
    "ActorCriticV": ActorCriticV,
    "ActorCriticT": ActorCriticT
}
class PPO:
    def __init__(self,
                 vec_env,
                 cfg_train,
                 cfg_env,
                 sampler='sequential',
                 log_dir='run',
                 is_testing=False,
                 print_log=True,
                 apply_reset=False,
                 device='cpu',
                 ):

        # if not isinstance(vec_env.observation_space, Space):
        #     raise TypeError("vec_env.observation_space must be a gym Space")
        # if not isinstance(vec_env.action_space, Space):
        #     raise TypeError("vec_env.action_space must be a gym Space")
        self.state_space = vec_env.observation_space
        self.action_space = vec_env.action_space

        self.cfg_train = copy.deepcopy(cfg_train)
        self.cfg_env = copy.deepcopy(cfg_env)
        learn_cfg = self.cfg_train["learn"]
        self.device = torch.device(device)

        self.desired_kl = learn_cfg.get("desired_kl", None)
        self.schedule = learn_cfg.get("schedule", "fixed")
        self.step_size = learn_cfg["optim_stepsize"]
        self.init_noise_std = learn_cfg.get("init_noise_std", 0.3)
        self.model_cfg = self.cfg_train["policy"]
        self.encoder_cfg = self.cfg_train['encoder']
        self.num_transitions_per_env=learn_cfg["nsteps"]
        # self.num_transitions_per_env = self.cfg_env['ep_length']  # do rollout each episode
        self.learning_rate=learn_cfg["optim_stepsize"]
        self.max_len=learn_cfg.get("max_len", 200)
        self.step_size_init = self.learning_rate

        # PPO components
        self.vec_env = vec_env
        self.actor_critic = eval(self.model_cfg['actor_critic'])(self.state_space.shape, self.action_space.shape, self.init_noise_std,
                                                                         self.model_cfg, self.encoder_cfg, self.cfg_env['env'])
        self.actor_critic.to(self.device)
        print(f"Encoder Name: {self.model_cfg['actor_critic']}")

        self.obs_device = self.device if self.encoder_cfg['name'] not in ['resnet18'] else 'cpu'
        self.storage = RolloutStorage(self.vec_env.num_envs, self.num_transitions_per_env, None, self.cfg_env['env']['obs_dim']['prop'],
                                      self.action_space.shape, self.device, sampler)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=self.learning_rate)

        # PPO parameters
        self.clip_param = learn_cfg["cliprange"]
        self.num_learning_epochs = learn_cfg["noptepochs"]
        self.num_mini_batches = learn_cfg["nminibatches"]
        # self.num_transitions_per_env = self.num_transitions_per_env
        self.value_loss_coef = learn_cfg.get("value_loss_coef", 2.0)
        self.entropy_coef = learn_cfg["ent_coef"]
        self.gamma = learn_cfg["gamma"]
        self.lam = learn_cfg["lam"]
        self.max_grad_norm = learn_cfg.get("max_grad_norm", 2.0)
        self.use_clipped_value_loss = learn_cfg.get("use_clipped_value_loss", False)
        self.model_dir = log_dir+'/checkpoint'
        os.makedirs(self.model_dir, exist_ok=True)

        # Log
        self.log_dir = log_dir+'/logger'
        self.print_log = print_log
        self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        self.tot_timesteps = 0
        self.tot_time = 0
        self.is_testing = is_testing
        self.current_learning_iteration = 0

        self.apply_reset = apply_reset

    def test(self, path):
        self.actor_critic.load_state_dict(torch.load(path, map_location=self.device))
        self.actor_critic.eval()

    def load(self, path):
        self.actor_critic.load_state_dict(torch.load(path, map_location=self.device))
        self.current_learning_iteration = int(path.split("_")[-1].split(".")[0])
        self.actor_critic.train()
    def load_from_dagger(self, path):
        parameters = torch.load(path, map_location=self.device)
        model_dict = self.actor_critic.state_dict()
        state_dict = {k:v for k,v in parameters.items() if 'obs_enc' not in k}
        model_dict.update(state_dict)

        self.actor_critic.load_state_dict(model_dict)

    def save(self, path):
        torch.save(self.actor_critic.state_dict(), path)

    def run(self, num_learning_iterations, log_interval=1):
        # current_obs_state = self.vec_env.reset()
        # current_states = self.vec_env.get_state()
        current_obs_state = self.vec_env.reset()
        if self.is_testing:
            maxlen = 100
            cur_reward_sum = torch.zeros(self.vec_env.num_envs, dtype=torch.float, device=self.device)
            cur_episode_length = torch.zeros(self.vec_env.num_envs, dtype=torch.float, device=self.device)

            reward_sum = []
            episode_length = []
            successes = []
            recoder = dict(images=[], tactiles=[])
            current_obs_state = self.vec_env.reset()
            while len(reward_sum) <= maxlen:
                # recoder["images"].append(current_obs_state[:, 50:-20].cpu().numpy().reshape(-1, 224, 224, 3))
                recoder["tactiles"].append(current_obs_state[:, -20:].cpu().numpy())
                with torch.no_grad():
                    # Compute the action
                    actions = self.actor_critic.act_inference(current_obs_state)
                    # Step the vec_environment
                    next_obs_state, rews, dones, infos = self.vec_env.step(actions)
                    # self.vec_env.render()
                    current_obs_state.copy_(next_obs_state)
                    cur_reward_sum[:] += rews
                    cur_episode_length[:] += 1

                    new_ids = (dones > 0).nonzero(as_tuple=False)
                    reward_sum.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                    episode_length.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                    successes.extend(infos["successes"][new_ids.cpu()][:, 0].cpu().numpy().tolist())
                    cur_reward_sum[new_ids] = 0
                    cur_episode_length[new_ids] = 0

                    if len(new_ids) > 0:
                        print("-" * 80)
                        print("Num episodes: {}".format(len(reward_sum)))
                        print("Mean return: {:.2f}".format(statistics.mean(reward_sum)))
                        print("Mean ep len: {:.2f}".format(statistics.mean(episode_length)))
                        print("Mean success: {:.2f}".format(statistics.mean(successes) * 100))
                        # pickle.dump(recoder, open(f'./I&T{len(reward_sum)}.pkl', 'wb'))
                        recoder = dict(images=[], tactiles=[])
                # print(f"sum of imges is {len(recoder['images'])}")

        else:
            rewbuffer = deque(maxlen=self.max_len)
            lenbuffer = deque(maxlen=self.max_len)
            successbuffer = deque(maxlen=self.max_len)
            # print(self.max_len)
            cur_reward_sum = torch.zeros(self.vec_env.num_envs, dtype=torch.float, device=self.device)
            cur_episode_length = torch.zeros(self.vec_env.num_envs, dtype=torch.float, device=self.device)
            reward_sum = []
            episode_length = []
            successes = []
            env_mean_success = 0.0

            for it in range(self.current_learning_iteration, num_learning_iterations):
                start = time.time()
                ep_infos = []


                # Rollout
                for _ in range(self.num_transitions_per_env):
                    if self.apply_reset:
                        current_obs = self.vec_env.reset()
                        current_states = self.vec_env.get_state()
                    # Compute the action
                    if current_obs_state.shape[0] > 500 and current_obs_state.shape[1] > 5000:
                        actions, actions_log_prob, current_obs_feats, current_state, mu, sigma, values = self.ac_act_split(current_obs_state)
                    else:
                        actions, actions_log_prob, values, mu, sigma, current_state, current_obs_feats = self.actor_critic.act(current_obs_state)
                    # Step the vec_environment
                    next_obs_state, rews, dones, infos = self.vec_env.step(actions)
                    # print(rews)
                    # next_states = self.vec_env.get_state()
                    # Record the transition
                    self.storage.add_transitions(current_state, current_obs_feats, actions, rews, dones, values, actions_log_prob, mu, sigma, self.obs_device)
                    current_obs_state.copy_(next_obs_state)
                    # current_states.copy_(next_states)
                    # Book keeping
                    ep_infos.append(infos)

                    if self.print_log:
                        cur_reward_sum[:] += rews
                        cur_episode_length[:] += 1

                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        # if len(new_ids)>0:
                        #     pass
                        reward_sum.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        episode_length.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        successes.extend(infos['successes'][new_ids.cpu()][:, 0].cpu().numpy().tolist())
                        # successes.extend(infos['env_mean_successes'][new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                        if "env_mean_successes" in infos.keys():
                            env_mean_success = infos['env_mean_successes'].mean()*100
                # sr statistics
                # all_sr = np.array(successes).reshape(self.num_transitions_per_env,self.vec_env.num_envs)
                # all_sr = np.sum(all_sr, axis=0) > 1

                if self.print_log:
                    # reward_sum = [x[0] for x in reward_sum]
                    # episode_length = [x[0] for x in episode_length]
                    rewbuffer.extend(reward_sum)
                    lenbuffer.extend(episode_length)
                    successbuffer.extend(successes)

                # _, _, last_values, _, _, _, _ = self.actor_critic.act(current_obs_state)
                if current_obs_state.shape[0] > 500 and current_obs_state.shape[1] > 5000:
                    _, _, _, _, _, _, last_values = self.ac_act_split(current_obs_state)
                else:
                    _, _, last_values, _, _, _, _ = self.actor_critic.act(current_obs_state)

                stop = time.time()
                collection_time = stop - start

                mean_trajectory_length, mean_reward = self.storage.get_statistics()

                # Learning step
                start = stop
                self.storage.compute_returns(last_values, self.gamma, self.lam)
                mean_value_loss, mean_surrogate_loss = self.update(it, num_learning_iterations)

                self.storage.clear()
                stop = time.time()
                learn_time = stop - start
                if self.print_log:
                    self.log(locals())
                if it % (log_interval/2) == 0:
                    # self.save(os.path.join(self.model_dir, 'model_{}.pt'.format(it)))
                    # # add 1129 for extract results in tensorboard online
                    self.results2cvs()
                if it % (2 * log_interval) == 0:
                    self.save(os.path.join(self.model_dir, 'model_{}.pt'.format(it)))
                ep_infos.clear()
            self.save(os.path.join(self.model_dir, 'model_{}.pt'.format(num_learning_iterations)))

    def ac_act_split(self, current_obs_state, max_image_bs = 200):
        # 获取 current_obs_state 的第一个维度
        batch_size = current_obs_state.shape[0]
        # 初始化存储结果的列表
        actions_list = []
        actions_log_prob_list = []
        values_list = []
        mu_list = []
        sigma_list = []
        current_state_list = []
        current_obs_feats_list = []
        # 如果 batch_size 大于 max_image_bs，将 current_obs_state 切分成多个 batch
        for i in range(0, batch_size, max_image_bs):
            batch_current_obs_state = current_obs_state[i:i + max_image_bs]

            # 对每个 batch 调用 act 方法
            actions, actions_log_prob, values, mu, sigma, current_state, current_obs_feats = self.actor_critic.act(
                batch_current_obs_state)

            # 将结果存储到列表中
            actions_list.append(actions)
            actions_log_prob_list.append(actions_log_prob)
            values_list.append(values)
            mu_list.append(mu)
            sigma_list.append(sigma)
            current_state_list.append(current_state)
            current_obs_feats_list.append(current_obs_feats)
        # 将列表中的张量拼接回原来的形状
        actions = torch.cat(actions_list, dim=0)
        actions_log_prob = torch.cat(actions_log_prob_list, dim=0)
        values = torch.cat(values_list, dim=0)
        mu = torch.cat(mu_list, dim=0)
        sigma = torch.cat(sigma_list, dim=0)
        current_state = torch.cat(current_state_list, dim=0)
        if None not in current_obs_feats_list:
            current_obs_feats = torch.cat(current_obs_feats_list, dim=0)
        else:
            current_obs_feats = None
        return actions, actions_log_prob, current_obs_feats, current_state, mu, sigma, values

    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_transitions_per_env * self.vec_env.num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']

        ep_string = f''
        # if locs['ep_infos']:
        #     for key in locs['ep_infos'][0]:
        #         infotensor = torch.tensor([], device=self.device)
        #         for ep_info in locs['ep_infos']:
        #             infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
        #         value = torch.mean(infotensor)
        #         self.writer.add_scalar('Episode/' + key, value, locs['it'])
        #         ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        mean_std = self.actor_critic.log_std.exp().mean()

        self.writer.add_scalar('Loss/value_function', locs['mean_value_loss'], locs['it'])
        self.writer.add_scalar('Loss/surrogate', locs['mean_surrogate_loss'], locs['it'])
        self.writer.add_scalar('Policy/mean_noise_std', mean_std.item(), locs['it'])
        self.writer.add_scalar('Train/env_mean_success', locs['env_mean_success'], locs['it'])
        if len(locs['rewbuffer']) > 0:
            self.writer.add_scalar('Train/mean_reward', statistics.mean(locs['rewbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_episode_length', statistics.mean(locs['lenbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_success', statistics.mean(locs['successbuffer'])*100, locs['it'])
            self.writer.add_scalar('Train/mean_reward/time', statistics.mean(locs['rewbuffer']), self.tot_time)
            self.writer.add_scalar('Train/mean_episode_length/time', statistics.mean(locs['lenbuffer']), self.tot_time)
            self.writer.add_scalar('Train/mean_success/time', statistics.mean(locs['successbuffer'])*100, self.tot_time)

        self.writer.add_scalar('Train2/mean_reward/step', locs['mean_reward'], locs['it'])
        self.writer.add_scalar('Train2/mean_episode_length/episode', locs['mean_trajectory_length'], locs['it'])

        fps = int(self.num_transitions_per_env * self.vec_env.num_envs / (locs['collection_time'] + locs['learn_time']))

        str = f" \033[1m Learning iteration {locs['it']}/{locs['num_learning_iterations']} \033[0m "

        if len(locs['rewbuffer']) > 0:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'task_name: ':>{pad}} {self.encoder_cfg['name']}\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                              'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                          f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                          f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
                          f"""{'Mean success rate:':>{pad}} {statistics.mean(locs['successbuffer'])*100:.2f}\n"""
                          f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                          f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
        else:
            log_string = (f"""{'#' * width}\n"""
                          
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'task_name: ':>{pad}} {self.encoder_cfg['name']}\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                          f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                          f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
        log_string += (f"""{'env_mean_success':>{pad}} {locs['env_mean_success']:.4f}\n""")
        log_string += ep_string
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                               locs['num_learning_iterations'] - locs['it']):.1f}s\n""")
        print(log_string)

    def results2cvs(self):

        out_path = os.path.dirname(self.log_dir) + "/log_train.csv"
        tensorboard2csv(self.log_dir, out_path)
        print(f"Generate results to {out_path}")

    def update(self, cur_iter, max_iter):
        mean_value_loss = 0
        mean_surrogate_loss = 0

        batch = self.storage.mini_batch_generator(self.num_mini_batches)
        for epoch in range(self.num_learning_epochs):
            # for obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch \
            #        in self.storage.mini_batch_generator(self.num_mini_batches):
            if self.schedule == "cos":
                self.step_size = self.adjust_learning_rate_cos(
                    self.optimizer, epoch, self.num_learning_epochs, cur_iter, max_iter
                )
            for indices in batch:
                obs_batch = self.storage.observations.view(-1, *self.storage.observations.size()[2:])[indices].to(self.device) if self.storage.observations is not None else None
                # if self.asymmetric:
                #     states_batch = self.storage.states.view(-1, *self.storage.states.size()[2:])[indices]
                # else:
                states_batch = self.storage.states.view(-1, *self.storage.states.size()[2:])[indices]
                actions_batch = self.storage.actions.view(-1, self.storage.actions.size(-1))[indices]
                target_values_batch = self.storage.values.view(-1, 1)[indices]
                returns_batch = self.storage.returns.view(-1, 1)[indices]
                old_actions_log_prob_batch = self.storage.actions_log_prob.view(-1, 1)[indices]
                advantages_batch = self.storage.advantages.view(-1, 1)[indices]
                old_mu_batch = self.storage.mu.view(-1, self.storage.actions.size(-1))[indices]
                old_sigma_batch = self.storage.sigma.view(-1, self.storage.actions.size(-1))[indices]

                actions_log_prob_batch, entropy_batch, value_batch, mu_batch, sigma_batch = self.actor_critic.evaluate(obs_batch, states_batch, actions_batch)

                # KL
                if self.desired_kl != None and self.schedule == 'adaptive':

                    kl = torch.sum(sigma_batch - old_sigma_batch + (torch.square(old_sigma_batch.exp()) + torch.square(old_mu_batch - mu_batch)) / (2.0 * torch.square(sigma_batch.exp())) - 0.5, axis=-1)
                    kl_mean = torch.mean(kl)

                    if kl_mean > self.desired_kl * 2.0:
                        self.step_size = max(1e-5, self.step_size / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.step_size = min(1e-2, self.step_size * 1.5)

                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = self.step_size

                # Surrogate loss
                ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
                surrogate = -torch.squeeze(advantages_batch) * ratio
                surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
                surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

                # Value function loss
                if self.use_clipped_value_loss:
                    value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param,
                                                                                                    self.clip_param)
                    value_losses = (value_batch - returns_batch).pow(2)
                    value_losses_clipped = (value_clipped - returns_batch).pow(2)
                    value_loss = torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = (returns_batch - value_batch).pow(2).mean()
                    # print(f'returns_batch: {returns_batch}')
                    # print(f'returns_batch: {returns_batch}')

                loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()
                # print(loss.item())
                # Gradient step
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

                mean_value_loss += value_loss.item()
                mean_surrogate_loss += surrogate_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates

        return mean_value_loss, mean_surrogate_loss

    def adjust_learning_rate_cos(self, optimizer, epoch, max_epoch, iter, max_iter):
        lr = self.step_size_init * 0.5 * (1. + math.cos(math.pi * (iter + epoch / max_epoch) / max_iter))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return lr

    def eval(self, logger, max_trajs=1000, maxlen=10e10, record_video=False):
        # maxlen = 200

        cur_reward_sum = torch.zeros(self.vec_env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.vec_env.num_envs, dtype=torch.float, device=self.device)

        reward_sum = []
        episode_length = []
        successes_que = []
        successes = dict()
        success_cnt = np.zeros(self.vec_env.num_envs)
        for i in range(self.vec_env.num_envs):
            successes[i] = list()

        current_obs_state = self.vec_env.reset()
        if record_video:
            video_frame_cnt = 0
            video_buf = self.vec_env.task.img_buf.cpu().unsqueeze(1).repeat(1, self.vec_env.task.max_episode_length, 1, 1, 1)
        force_seq = []
        while len(reward_sum) < maxlen:
            with torch.no_grad():
                # Compute the action
                actions = self.actor_critic.act_inference(current_obs_state)
                # Step the vec_environment
                next_obs_state, rews, dones, infos = self.vec_env.step(actions)
                # force_seq.append(self.vec_env.task.force)
                # time.sleep(0.01)
                # self.vec_env.render()
                current_obs_state.copy_(next_obs_state)
                cur_reward_sum[:] += rews
                cur_episode_length[:] += 1

                new_ids = (dones > 0).nonzero(as_tuple=False)
                reward_sum.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                episode_length.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                successes_que.extend(infos["successes"][new_ids.cpu()][:, 0].cpu().numpy().tolist())
                for item in new_ids[:, 0]:
                    successes[item.item()].append(infos["successes"][item].item())
                    success_cnt[item.item()] += 1

                cur_reward_sum[new_ids] = 0
                cur_episode_length[new_ids] = 0

                # record videos
                if record_video:
                    if video_frame_cnt < self.vec_env.task.max_episode_length:
                        self.vec_env.task.compute_pixel_obs()
                        video_buf[:, video_frame_cnt, :, :] = self.vec_env.task.img_buf.cpu()
                        video_frame_cnt += 1
                    else:
                        video_save_path = os.path.dirname(self.log_dir) + self.cfg_env["video_path"]
                        os.makedirs(video_save_path, exist_ok=True)
                        for i in range(self.cfg_env["env"]["numEnvs"]):
                            write_video(video_save_path+f"/{self.cfg_env['env']['env_dict'][i//len(self.cfg_env['env']['env_dict'])]}_{i}.mp4", video_buf[i, ...], 30)
                            print(f"Video saved in {video_save_path}+{self.cfg_env['env']['env_dict'][i//len(self.cfg_env['env']['env_dict'])]}_{i}.mp4")
                        record_video = False
                # if force_seq.__len__()>=1000:
                #     save_path = os.path.dirname(self.log_dir) + '/force_seq'
                #     os.makedirs(save_path, exist_ok=True)
                #     np.save(save_path+"/force.npy", force_seq)
                #     break
                if len(new_ids) > 0:
                    from utils.util import plot_tensor_data
                    # plot_tensor_data(self.vec_env.task.action_penalty, "vt-abs")
                    # plot_tensor_data(self.vec_env.task.action_penalty1, "vt-rela")
                    print("-" * 80)
                    # print("Num episodes: {}".format(len(reward_sum)))
                    # print("Mean return: {:.2f}".format(statistics.mean(reward_sum)))
                    # print("Mean ep len: {:.2f}".format(statistics.mean(episode_length)))
                    print(f"Mean env_success: {self.vec_env.task.env_mean_successes.mean(-1)}")
                    print("Total Num trajs: {}".format(len(successes_que)))
                    print(f"Mean success: {statistics.mean(successes_que)}")
                    # if np.min(success_cnt)>0:
                    #     mean_env_success = np.zeros(self.vec_env.num_envs)
                    #     for key, value in successes.items():
                    #         mean_env_success[key] = np.mean(value[:max_trajs])
                    #     total_mean_sr = mean_env_success.mean()
                    #     print("Mean success: {:.2f}".format(total_mean_sr * 100))
                    #
                    # if np.min(success_cnt) == max_trajs:
                    #     for i, name in enumerate(self.cfg_env["env"]["env_dict"]):
                    #         logger.log_kv(name, mean_env_success[i]*100)
                    #     logger.log_kv("total_mean", total_mean_sr*100)
                    # break
                    if len(successes_que) > self.vec_env.num_envs * max_trajs:
                        logger.log_kv("total_mean", statistics.mean(successes_que) * 100)
                        break

        # return statistics.mean(reward_sum), statistics.mean(successes)

    def collect_demos(self, maxlen, task_name):
        # maxlen = 200
        import cv2
        cur_reward_sum = torch.zeros(self.vec_env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.vec_env.num_envs, dtype=torch.float, device=self.device)

        reward_sum = []
        episode_length = []
        successes = []

        states = [[] for _ in range(self.vec_env.num_envs)]
        obs = [[] for _ in range(self.vec_env.num_envs)]
        actions = [[] for _ in range(self.vec_env.num_envs)]
        rewards = [[] for _ in range(self.vec_env.num_envs)]
        qpos = [[] for _ in range(self.vec_env.num_envs)]
        trajs = []

        t = 0
        current_obs_state = self.vec_env.reset()
        # while len(successes) < maxlen:
        while t < maxlen:
            with torch.no_grad():
                # Compute the action
                action = self.actor_critic.act_inference(current_obs_state)
                _, actions_log_prob, values, mu, sigma, current_state, current_obs_feats = self.actor_critic.act(
                    current_obs_state)

                # Step the vec_environment
                next_obs_state, rews, dones, infos = self.vec_env.step(action)

                q = self.vec_env.task.shadow_hand_dof_pos

                # self.vec_env.render()
                current_obs_state.copy_(next_obs_state)
                cur_reward_sum[:] += rews
                cur_episode_length[:] += 1

                new_ids = (dones > 0).nonzero(as_tuple=False)
                reward_sum.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                episode_length.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                infos["successes"] = infos["successes"].to(self.device)
                successes.extend(infos["successes"][new_ids][:, 0].cpu().numpy().tolist())

                cur_reward_sum[new_ids] = 0
                cur_episode_length[new_ids] = 0

                if len(new_ids) > 0:
                    print("-" * 80)
                    print("Num episodes: {}".format(len(reward_sum)))
                    print("Mean return: {:.2f}".format(statistics.mean(reward_sum)))
                    print("Mean ep len: {:.2f}".format(statistics.mean(episode_length)))
                    print("Mean success: {:.2f}".format(statistics.mean(successes) * 100))

                # for i in range(action.shape[0]):
                # image = np.array(self.vec_env.task.render_buf[i].cpu())
                # # image = cv2.flip(image, 0)
                # os.makedirs(traj_saved_path + '/traj_%d' % traj_index + '/obj_%d' % i, exist_ok=True)
                # cv2.imwrite(traj_saved_path + '/traj_%d/obj_%d/%d.jpg' % (traj_index, i, t), image)

                for i in range(self.vec_env.num_envs):
                    states[i].append(np.array(current_state[i].cpu()))
                    obs[i].append(np.array(current_obs_feats[i].cpu()))
                    actions[i].append(np.array(action[i].cpu()))
                    rewards[i].append(np.array(rews[i].cpu()))
                    qpos[i].append(np.array(q[i].cpu()))
                    collect = copy.deepcopy(infos['successes'])[i].cpu().numpy()
                    if collect:
                        traj = dict(
                            task_name=task_name,
                            states=np.array(states[i]),
                            obs=np.array(obs[i]),
                            actions=np.array(actions[i]),
                            rewards=np.array(rewards[i]),
                            qpos=np.array(qpos[i]),
                        )
                        trajs.append(traj)

                t += 1

        # return statistics.mean(reward_sum), statistics.mean(successes), traj
        return trajs


