import os

import numpy as np

import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from model.backbones.pre_model import Encoder, Encoder2, Encoder_T, Encoder_no_pre, EncoderVE_T, EncoderV_no_pre
import torchvision.models as models

class ActorCritic(nn.Module):

    def __init__(self, obs_shape, actions_shape, initial_std, model_cfg, encoder_cfg, env_cfg):
        super(ActorCritic, self).__init__()

        # Encoder
        # emb_dim = encoder_cfg["emb_dim"]
        # self.state_enc = nn.Linear(env_cfg['obs_dim']['prop'], emb_dim)
        self.obs_dim = [v for v in env_cfg['obs_dim'].values()]
        if model_cfg is None:
            actor_hidden_dim = [256, 256, 256]
            critic_hidden_dim = [256, 256, 256]
            activation = get_activation("selu")
        else:
            actor_hidden_dim = model_cfg['pi_hid_sizes']
            critic_hidden_dim = model_cfg['vf_hid_sizes']
            activation = get_activation(model_cfg['activation'])

        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(*obs_shape, actor_hidden_dim[0]))
        actor_layers.append(activation)
        for l in range(len(actor_hidden_dim)):
            if l == len(actor_hidden_dim) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dim[l], *actions_shape))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dim[l], actor_hidden_dim[l + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)

        # Value function
        critic_layers = []

        critic_layers.append(nn.Linear(*obs_shape, critic_hidden_dim[0]))
        critic_layers.append(activation)
        for l in range(len(critic_hidden_dim)):
            if l == len(critic_hidden_dim) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dim[l], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dim[l], critic_hidden_dim[l + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        # print(self.actor)
        # print(self.critic)

        # Action noise
        self.log_std = nn.Parameter(np.log(initial_std) * torch.ones(*actions_shape))

        # Initialize the weights like in stable baselines
        actor_weights = [np.sqrt(2)] * len(actor_hidden_dim)
        actor_weights.append(0.01)
        critic_weights = [np.sqrt(2)] * len(critic_hidden_dim)
        critic_weights.append(1.0)
        self.init_weights(self.actor, actor_weights)
        self.init_weights(self.critic, critic_weights)

    @staticmethod
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

    def forward(self):
        raise NotImplementedError

    @torch.no_grad()
    def act(self, state):
        # state_emb = self.state_enc(state)
        actions_mean = self.actor(state)

        covariance = torch.diag(self.log_std.exp() * self.log_std.exp())
        distribution = MultivariateNormal(actions_mean, scale_tril=covariance)

        actions = distribution.sample()
        actions_log_prob = distribution.log_prob(actions)

        value = self.critic(state)

        return actions.detach(), actions_log_prob.detach(), value.detach(), actions_mean.detach(), self.log_std.repeat(actions_mean.shape[0], 1).detach(), state, None

    @torch.no_grad()
    def act_inference(self, observations):
        # state_emb = self.state_enc(observations)
        actions_mean = self.actor(observations)
        return actions_mean

    def evaluate(self, observations, state, actions):
        # state_emb = self.state_enc(state)
        actions_mean = self.actor(state)

        covariance = torch.diag(self.log_std.exp() * self.log_std.exp())
        distribution = MultivariateNormal(actions_mean, scale_tril=covariance)

        actions_log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()


        value = self.critic(state)

        return actions_log_prob, entropy, value, actions_mean, self.log_std.repeat(actions_mean.shape[0], 1)

class ActorCriticVEncoder(nn.Module):

    def __init__(self, obs_shape, actions_shape, initial_std, model_cfg, encoder_cfg, env_cfg):
        super(ActorCriticVEncoder, self).__init__()

        self.obs_dim = [v for v in env_cfg['obs_dim'].values()]
        # Encoder
        emb_dim = encoder_cfg["emb_dim"]

        self.obs_enc = Encoder(
            model_name=encoder_cfg["name"],
            pretrain_dir=encoder_cfg["pretrain_dir"],
            freeze=encoder_cfg["freeze"],
            emb_dim=emb_dim,
            en_mode=encoder_cfg["en_mode"],
            f_ex_mode=encoder_cfg["f_ex_mode"]
        )

        self.state_enc = nn.Linear(env_cfg['obs_dim']['prop'], emb_dim)
        # self.obs_fc = nn.Linear(64, obs_emb)

        if model_cfg is None:
            actor_hidden_dim = [256, 256, 256]
            critic_hidden_dim = [256, 256, 256]
            activation = get_activation("selu")
        else:
            actor_hidden_dim = model_cfg['pi_hid_sizes']
            critic_hidden_dim = model_cfg['vf_hid_sizes']
            activation = get_activation(model_cfg['activation'])

        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(len(self.obs_dim)*emb_dim, actor_hidden_dim[0]))
        actor_layers.append(activation)
        for l in range(len(actor_hidden_dim)):
            if l == len(actor_hidden_dim) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dim[l], *actions_shape))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dim[l], actor_hidden_dim[l + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)

        # Value function
        # self.extractors_vf = nn.ModuleList(LinearEncoder(self.obs_dim[i], hidden_size, obs_emb) for i in range(self.obs_dim.__len__()))
        critic_layers = []

        critic_layers.append(nn.Linear(len(self.obs_dim)*emb_dim, critic_hidden_dim[0]))
        critic_layers.append(activation)
        for l in range(len(critic_hidden_dim)):
            if l == len(critic_hidden_dim) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dim[l], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dim[l], critic_hidden_dim[l + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        # print(self.obs_enc)
        # print(self.actor)
        # print(self.critic)

        # Action noise
        self.log_std = nn.Parameter(np.log(initial_std) * torch.ones(*actions_shape))

        # Initialize the weights like in stable baselines
        actor_weights = [np.sqrt(2)] * len(actor_hidden_dim)
        actor_weights.append(0.01)
        critic_weights = [np.sqrt(2)] * len(critic_hidden_dim)
        critic_weights.append(1.0)
        self.init_weights(self.actor, actor_weights)
        self.init_weights(self.critic, critic_weights)

    @staticmethod
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

    def obs_division(self, x):
        n_input_types = len(self.obs_dim)
        assert n_input_types > 1
        x_list = []
        st_idx = 0
        for idx in range(n_input_types):
            end_idx = st_idx + self.obs_dim[idx]
            x_list.append(x[:, st_idx:end_idx])
            st_idx += self.obs_dim[idx]

        return x_list
    # def extract_feat(self, x):
    #
    #     return o_en, state, obs_feat

    def forward(self):
        raise NotImplementedError

    @torch.no_grad()
    def act(self, observations):
        state, obs = self.obs_division(observations)
        state_emb = self.state_enc(state)
        obs =obs.view(-1, 224, 224, 3).to(torch.uint8) #image
        obs_emb, obs_feat = self.obs_enc(obs.permute(0, 3, 1, 2))
        joint_emb = torch.cat([state_emb, obs_emb], dim=1)

        actions_mean = self.actor(joint_emb)

        covariance = torch.diag(self.log_std.exp() * self.log_std.exp())
        distribution = MultivariateNormal(actions_mean, scale_tril=covariance)

        actions = distribution.sample()
        actions_log_prob = distribution.log_prob(actions)

        # obs_emb_vf = self.extract_feat(observations, self.extractors_vf)
        value = self.critic(joint_emb)

        return actions.detach(), \
               actions_log_prob.detach(), \
               value.detach(), \
               actions_mean.detach(), \
               self.log_std.repeat(actions_mean.shape[0], 1).detach(), \
               state.detach(),\
               obs_feat.detach()

    @torch.no_grad()
    def act_inference(self, observations):
        state, obs = self.obs_division(observations)
        state_emb = self.state_enc(state)
        obs = obs.view(-1, 224, 224, 3).to(torch.uint8)  # image
        obs_emb, obs_feat = self.obs_enc(obs.permute(0, 3, 1, 2))
        joint_emb = torch.cat([state_emb, obs_emb], dim=1)

        actions_mean = self.actor(joint_emb)
        return actions_mean

    def evaluate(self, obs_features, state, actions):

        state_emb = self.state_enc(state)
        obs_emb = self.obs_enc.forward_feat(obs_features)
        joint_emb = torch.cat([state_emb, obs_emb], dim=1)
        actions_mean = self.actor(joint_emb)

        covariance = torch.diag(self.log_std.exp() * self.log_std.exp())
        distribution = MultivariateNormal(actions_mean, scale_tril=covariance)

        actions_log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()

        # obs_emb_vf = self.extract_feat(observations, self.extractors_vf)
        value = self.critic(joint_emb)

        return actions_log_prob, entropy, value, actions_mean, self.log_std.repeat(actions_mean.shape[0], 1)
class ActorCriticTEncoder(nn.Module):

    def __init__(self, obs_shape, actions_shape, initial_std, model_cfg, encoder_cfg, env_cfg):
        super(ActorCriticTEncoder, self).__init__()

        self.obs_dim = [v for v in env_cfg['obs_dim'].values()]
        # Encoder
        emb_dim = encoder_cfg["emb_dim"]

        self.obs_enc = Encoder_T(
            model_name=encoder_cfg["name"],
            pretrain_dir=encoder_cfg["pretrain_dir"],
            freeze=encoder_cfg["freeze"],
            emb_dim=emb_dim,
            en_mode=encoder_cfg["en_mode"],
            f_ex_mode=encoder_cfg["f_ex_mode"]

        )

        self.state_enc = nn.Linear(env_cfg['obs_dim']['prop'], emb_dim)
        # self.obs_fc = nn.Linear(64, obs_emb)

        if model_cfg is None:
            actor_hidden_dim = [256, 256, 256]
            critic_hidden_dim = [256, 256, 256]
            activation = get_activation("selu")
        else:
            actor_hidden_dim = model_cfg['pi_hid_sizes']
            critic_hidden_dim = model_cfg['vf_hid_sizes']
            activation = get_activation(model_cfg['activation'])

        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(len(self.obs_dim)*emb_dim, actor_hidden_dim[0]))
        actor_layers.append(activation)
        for l in range(len(actor_hidden_dim)):
            if l == len(actor_hidden_dim) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dim[l], *actions_shape))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dim[l], actor_hidden_dim[l + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)

        # Value function
        # self.extractors_vf = nn.ModuleList(LinearEncoder(self.obs_dim[i], hidden_size, obs_emb) for i in range(self.obs_dim.__len__()))
        critic_layers = []

        critic_layers.append(nn.Linear(len(self.obs_dim)*emb_dim, critic_hidden_dim[0]))
        critic_layers.append(activation)
        for l in range(len(critic_hidden_dim)):
            if l == len(critic_hidden_dim) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dim[l], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dim[l], critic_hidden_dim[l + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        # print(self.obs_enc)
        # print(self.actor)
        # print(self.critic)

        # Action noise
        self.log_std = nn.Parameter(np.log(initial_std) * torch.ones(*actions_shape))

        # Initialize the weights like in stable baselines
        actor_weights = [np.sqrt(2)] * len(actor_hidden_dim)
        actor_weights.append(0.01)
        critic_weights = [np.sqrt(2)] * len(critic_hidden_dim)
        critic_weights.append(1.0)
        self.init_weights(self.actor, actor_weights)
        self.init_weights(self.critic, critic_weights)

    @staticmethod
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

    def obs_division(self, x):
        n_input_types = len(self.obs_dim)
        assert n_input_types > 1
        x_list = []
        st_idx = 0
        for idx in range(n_input_types):
            end_idx = st_idx + self.obs_dim[idx]
            x_list.append(x[:, st_idx:end_idx])
            st_idx += self.obs_dim[idx]

        return x_list

    # def extract_feat(self, x):
    #
    #     return o_en, state, obs_feat

    def forward(self):
        raise NotImplementedError

    @torch.no_grad()
    def act(self, observations):
        state, obs = self.obs_division(observations)
        state_emb = self.state_enc(state)
        # obs = obs.view(-1, 224, 224, 3).to(torch.uint8)  # image
        obs_emb, obs_feat = self.obs_enc(obs)
        joint_emb = torch.cat([state_emb, obs_emb], dim=1)

        actions_mean = self.actor(joint_emb)

        covariance = torch.diag(self.log_std.exp() * self.log_std.exp())
        distribution = MultivariateNormal(actions_mean, scale_tril=covariance)

        actions = distribution.sample()
        actions_log_prob = distribution.log_prob(actions)

        # obs_emb_vf = self.extract_feat(observations, self.extractors_vf)
        value = self.critic(joint_emb)

        return actions.detach(), \
               actions_log_prob.detach(), \
               value.detach(), \
               actions_mean.detach(), \
               self.log_std.repeat(actions_mean.shape[0], 1).detach(), \
               state.detach(), \
               obs_feat.detach()

    @torch.no_grad()
    def act_inference(self, observations):
        state, obs = self.obs_division(observations)
        state_emb = self.state_enc(state)
        # obs = obs.view(-1, 224, 224, 3).to(torch.uint8)  # image
        obs_emb, obs_feat = self.obs_enc(obs)
        joint_emb = torch.cat([state_emb, obs_emb], dim=1)

        actions_mean = self.actor(joint_emb)
        return actions_mean

    def evaluate(self, obs_features, state, actions):

        state_emb = self.state_enc(state)
        obs_emb = self.obs_enc.forward_feat(obs_features)
        joint_emb = torch.cat([state_emb, obs_emb], dim=1)
        actions_mean = self.actor(joint_emb)

        covariance = torch.diag(self.log_std.exp() * self.log_std.exp())
        distribution = MultivariateNormal(actions_mean, scale_tril=covariance)

        actions_log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()

        # obs_emb_vf = self.extract_feat(observations, self.extractors_vf)
        value = self.critic(joint_emb)

        return actions_log_prob, entropy, value, actions_mean, self.log_std.repeat(actions_mean.shape[0], 1)
class ActorCriticVTEncoder(nn.Module):

    def __init__(self, obs_shape, actions_shape, initial_std, model_cfg, encoder_cfg, env_cfg):
        super(ActorCriticVTEncoder, self).__init__()

        self.obs_dim = [v for v in env_cfg['obs_dim'].values()]
        # Encoder
        emb_dim = encoder_cfg["emb_dim"]

        self.obs_enc = Encoder(
            model_name=encoder_cfg["name"],
            pretrain_dir=encoder_cfg["pretrain_dir"],
            freeze=encoder_cfg["freeze"],
            emb_dim=emb_dim,
            en_mode=encoder_cfg["en_mode"],
            f_ex_mode=encoder_cfg["f_ex_mode"]
        )

        self.state_enc = nn.Linear(env_cfg['obs_dim']['prop'], emb_dim)
        # self.obs_fc = nn.Linear(64, obs_emb)

        if model_cfg is None:
            actor_hidden_dim = [256, 256, 256]
            critic_hidden_dim = [256, 256, 256]
            activation = get_activation("selu")
        else:
            actor_hidden_dim = model_cfg['pi_hid_sizes']
            critic_hidden_dim = model_cfg['vf_hid_sizes']
            activation = get_activation(model_cfg['activation'])

        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(2*emb_dim, actor_hidden_dim[0]))
        actor_layers.append(activation)
        for l in range(len(actor_hidden_dim)):
            if l == len(actor_hidden_dim) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dim[l], *actions_shape))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dim[l], actor_hidden_dim[l + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)

        # Value function
        # self.extractors_vf = nn.ModuleList(LinearEncoder(self.obs_dim[i], hidden_size, obs_emb) for i in range(self.obs_dim.__len__()))
        critic_layers = []

        critic_layers.append(nn.Linear(2*emb_dim, critic_hidden_dim[0]))
        critic_layers.append(activation)
        for l in range(len(critic_hidden_dim)):
            if l == len(critic_hidden_dim) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dim[l], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dim[l], critic_hidden_dim[l + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        # print(self.obs_enc)
        # print(self.actor)
        # print(self.critic)

        # Action noise
        self.log_std = nn.Parameter(np.log(initial_std) * torch.ones(*actions_shape))

        # Initialize the weights like in stable baselines
        actor_weights = [np.sqrt(2)] * len(actor_hidden_dim)
        actor_weights.append(0.01)
        critic_weights = [np.sqrt(2)] * len(critic_hidden_dim)
        critic_weights.append(1.0)
        self.init_weights(self.actor, actor_weights)
        self.init_weights(self.critic, critic_weights)

    @staticmethod
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

    def obs_division(self, x):
        n_input_types = len(self.obs_dim)
        assert n_input_types > 1
        x_list = []
        st_idx = 0
        for idx in range(n_input_types):
            end_idx = st_idx + self.obs_dim[idx]
            x_list.append(x[:, st_idx:end_idx])
            st_idx += self.obs_dim[idx]

        return x_list
    # def extract_feat(self, x):
    #
    #     return o_en, state, obs_feat

    def forward(self):
        raise NotImplementedError

    @torch.no_grad()
    def act(self, observations):
        state, img, tac = self.obs_division(observations)
        state_emb = self.state_enc(state)
        img = img.view(-1, 224, 224, 3).to(torch.uint8)  # image
        obs_emb, obs_feat = self.obs_enc((img.permute(0, 3, 1, 2), tac))
        joint_emb = torch.cat([state_emb, obs_emb], dim=1)


        actions_mean = self.actor(joint_emb)

        covariance = torch.diag(self.log_std.exp() * self.log_std.exp())
        try:
            distribution = MultivariateNormal(actions_mean, scale_tril=covariance)
        except:
            print(f"log_std: {self.log_std}")
            print(f"actions_mean: {actions_mean}")

        actions = distribution.sample()
        actions_log_prob = distribution.log_prob(actions)

        # obs_emb_vf = self.extract_feat(observations, self.extractors_vf)
        value = self.critic(joint_emb)

        return actions.detach(), \
               actions_log_prob.detach(), \
               value.detach(), \
               actions_mean.detach(), \
               self.log_std.repeat(actions_mean.shape[0], 1).detach(), \
               state.detach(),\
               obs_feat.detach()

    @torch.no_grad()
    def act_inference(self, observations):
        state, img, tac = self.obs_division(observations)
        state_emb = self.state_enc(state)
        img = img.view(-1, 224, 224, 3).to(torch.uint8)  # image
        obs_emb, obs_feat = self.obs_enc((img.permute(0, 3, 1, 2), tac))
        joint_emb = torch.cat([state_emb, obs_emb], dim=1)

        actions_mean = self.actor(joint_emb)
        return actions_mean

    def evaluate(self, obs_features, state, actions):

        state_emb = self.state_enc(state)
        obs_emb = self.obs_enc.forward_feat(obs_features)
        joint_emb = torch.cat([state_emb, obs_emb], dim=1)
        actions_mean = self.actor(joint_emb)

        covariance = torch.diag(self.log_std.exp() * self.log_std.exp())
        distribution = MultivariateNormal(actions_mean, scale_tril=covariance)

        actions_log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()

        # obs_emb_vf = self.extract_feat(observations, self.extractors_vf)
        value = self.critic(joint_emb)

        return actions_log_prob, entropy, value, actions_mean, self.log_std.repeat(actions_mean.shape[0], 1)
class ActorCriticVT(nn.Module):

    def __init__(self, obs_shape, actions_shape, initial_std, model_cfg, encoder_cfg, env_cfg):
        super(ActorCriticVT, self).__init__()

        self.obs_dim = [v for v in env_cfg['obs_dim'].values()]
        # Encoder
        emb_dim = encoder_cfg["emb_dim"]
        if encoder_cfg["name"] == "resnet18":
            self.obs_enc = Encoder_no_pre(
                model_name=encoder_cfg["name"],
                emb_dim=emb_dim,
            )
        else:
            raise AssertionError(f"There is no defined model {env_cfg['name']}")

        self.state_enc = nn.Linear(env_cfg['obs_dim']['prop'], emb_dim)
        # self.obs_fc = nn.Linear(64, obs_emb)

        if model_cfg is None:
            actor_hidden_dim = [256, 256, 256]
            critic_hidden_dim = [256, 256, 256]
            activation = get_activation("selu")
        else:
            actor_hidden_dim = model_cfg['pi_hid_sizes']
            critic_hidden_dim = model_cfg['vf_hid_sizes']
            activation = get_activation(model_cfg['activation'])

        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(3*emb_dim, actor_hidden_dim[0]))
        actor_layers.append(activation)
        for l in range(len(actor_hidden_dim)):
            if l == len(actor_hidden_dim) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dim[l], *actions_shape))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dim[l], actor_hidden_dim[l + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)

        # Value function
        # self.extractors_vf = nn.ModuleList(LinearEncoder(self.obs_dim[i], hidden_size, obs_emb) for i in range(self.obs_dim.__len__()))
        critic_layers = []

        critic_layers.append(nn.Linear(3*emb_dim, critic_hidden_dim[0]))
        critic_layers.append(activation)
        for l in range(len(critic_hidden_dim)):
            if l == len(critic_hidden_dim) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dim[l], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dim[l], critic_hidden_dim[l + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        # print(self.obs_enc)
        # print(self.actor)
        # print(self.critic)

        # Action noise
        self.log_std = nn.Parameter(np.log(initial_std) * torch.ones(*actions_shape))

        # Initialize the weights like in stable baselines
        actor_weights = [np.sqrt(2)] * len(actor_hidden_dim)
        actor_weights.append(0.01)
        critic_weights = [np.sqrt(2)] * len(critic_hidden_dim)
        critic_weights.append(1.0)
        self.init_weights(self.actor, actor_weights)
        self.init_weights(self.critic, critic_weights)

    @staticmethod
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

    def obs_division(self, x):
        n_input_types = len(self.obs_dim)
        assert n_input_types > 1
        x_list = []
        st_idx = 0
        for idx in range(n_input_types):
            end_idx = st_idx + self.obs_dim[idx]
            x_list.append(x[:, st_idx:end_idx])
            st_idx += self.obs_dim[idx]

        return x_list
    # def extract_feat(self, x):
    #
    #     return o_en, state, obs_feat

    def forward(self):
        raise NotImplementedError

    @torch.no_grad()
    def act(self, observations):
        self.obs_enc.eval()
        state, img, tac = self.obs_division(observations)
        state_emb = self.state_enc(state)
        img = img.view(-1, 224, 224, 3).to(torch.uint8)  # image
        obs_emb = self.obs_enc((img.permute(0, 3, 1, 2), tac))
        joint_emb = torch.cat([state_emb, obs_emb], dim=1)


        actions_mean = self.actor(joint_emb)

        covariance = torch.diag(self.log_std.exp() * self.log_std.exp())
        distribution = MultivariateNormal(actions_mean, scale_tril=covariance)

        actions = distribution.sample()
        actions_log_prob = distribution.log_prob(actions)

        # obs_emb_vf = self.extract_feat(observations, self.extractors_vf)
        value = self.critic(joint_emb)
        self.obs_enc.train()

        return actions.detach(), \
               actions_log_prob.detach(), \
               value.detach(), \
               actions_mean.detach(), \
               self.log_std.repeat(actions_mean.shape[0], 1).detach(), \
               state.detach(),\
               observations[:, state.shape[1]:].detach()

    @torch.no_grad()
    def act_inference(self, observations):
        self.obs_enc.eval()
        state, img, tac = self.obs_division(observations)
        state_emb = self.state_enc(state)
        img = img.view(-1, 224, 224, 3).to(torch.uint8)  # image
        obs_emb = self.obs_enc((img.permute(0, 3, 1, 2), tac))
        joint_emb = torch.cat([state_emb, obs_emb], dim=1)

        actions_mean = self.actor(joint_emb)
        return actions_mean

    def evaluate(self, obs_features, state, actions):

        state_emb = self.state_enc(state)
        obs_emb = self.obs_enc.forward(obs_features)
        joint_emb = torch.cat([state_emb, obs_emb], dim=1)
        actions_mean = self.actor(joint_emb)

        covariance = torch.diag(self.log_std.exp() * self.log_std.exp())
        distribution = MultivariateNormal(actions_mean, scale_tril=covariance)

        actions_log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()

        # obs_emb_vf = self.extract_feat(observations, self.extractors_vf)
        value = self.critic(joint_emb)

        return actions_log_prob, entropy, value, actions_mean, self.log_std.repeat(actions_mean.shape[0], 1)
class ActorCriticV_TEncoder(nn.Module):

    def __init__(self, obs_shape, actions_shape, initial_std, model_cfg, encoder_cfg, env_cfg):
        super(ActorCriticV_TEncoder, self).__init__()

        self.obs_dim = [v for v in env_cfg['obs_dim'].values()]
        # Encoder
        emb_dim = encoder_cfg["emb_dim"]

        self.obs_enc = Encoder2(
            model_name=encoder_cfg["name"],
            pretrain_dir=encoder_cfg["pretrain_dir"],
            freeze=encoder_cfg["freeze"],
            emb_dim=emb_dim,
            en_mode=encoder_cfg["en_mode"]
        )

        self.state_enc = nn.Linear(env_cfg['obs_dim']['prop'], emb_dim)
        # self.obs_fc = nn.Linear(64, obs_emb)

        if model_cfg is None:
            actor_hidden_dim = [256, 256, 256]
            critic_hidden_dim = [256, 256, 256]
            activation = get_activation("selu")
        else:
            actor_hidden_dim = model_cfg['pi_hid_sizes']
            critic_hidden_dim = model_cfg['vf_hid_sizes']
            activation = get_activation(model_cfg['activation'])

        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(3*emb_dim, actor_hidden_dim[0]))
        actor_layers.append(activation)
        for l in range(len(actor_hidden_dim)):
            if l == len(actor_hidden_dim) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dim[l], *actions_shape))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dim[l], actor_hidden_dim[l + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)

        # Value function
        # self.extractors_vf = nn.ModuleList(LinearEncoder(self.obs_dim[i], hidden_size, obs_emb) for i in range(self.obs_dim.__len__()))
        critic_layers = []

        critic_layers.append(nn.Linear(3*emb_dim, critic_hidden_dim[0]))
        critic_layers.append(activation)
        for l in range(len(critic_hidden_dim)):
            if l == len(critic_hidden_dim) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dim[l], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dim[l], critic_hidden_dim[l + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        # print(self.obs_enc)
        # print(self.actor)
        # print(self.critic)

        # Action noise
        self.log_std = nn.Parameter(np.log(initial_std) * torch.ones(*actions_shape))

        # Initialize the weights like in stable baselines
        actor_weights = [np.sqrt(2)] * len(actor_hidden_dim)
        actor_weights.append(0.01)
        critic_weights = [np.sqrt(2)] * len(critic_hidden_dim)
        critic_weights.append(1.0)
        self.init_weights(self.actor, actor_weights)
        self.init_weights(self.critic, critic_weights)
    @staticmethod
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

    def obs_division(self, x):
        n_input_types = len(self.obs_dim)
        assert n_input_types > 1
        x_list = []
        st_idx = 0
        for idx in range(n_input_types):
            end_idx = st_idx + self.obs_dim[idx]
            x_list.append(x[:, st_idx:end_idx])
            st_idx += self.obs_dim[idx]

        return x_list
    # def extract_feat(self, x):
    #
    #     return o_en, state, obs_feat

    def forward(self):
        raise NotImplementedError

    @torch.no_grad()
    def act(self, observations):
        self.obs_enc.eval()
        state, img, tac = self.obs_division(observations)
        state_emb = self.state_enc(state)
        img = img.view(-1, 224, 224, 3).to(torch.uint8)  # image
        obs_emb, obs_feat = self.obs_enc((img.permute(0, 3, 1, 2), tac))
        joint_emb = torch.cat([state_emb, obs_emb], dim=1)


        actions_mean = self.actor(joint_emb)

        covariance = torch.diag(self.log_std.exp() * self.log_std.exp())
        distribution = MultivariateNormal(actions_mean, scale_tril=covariance)

        actions = distribution.sample()
        actions_log_prob = distribution.log_prob(actions)

        # obs_emb_vf = self.extract_feat(observations, self.extractors_vf)
        value = self.critic(joint_emb)
        self.obs_enc.train()

        return actions.detach(), \
               actions_log_prob.detach(), \
               value.detach(), \
               actions_mean.detach(), \
               self.log_std.repeat(actions_mean.shape[0], 1).detach(), \
               state.detach(),\
               obs_feat.detach()

    @torch.no_grad()
    def act_inference(self, observations):
        self.obs_enc.eval()
        state, img, tac = self.obs_division(observations)
        state_emb = self.state_enc(state)
        img = img.view(-1, 224, 224, 3).to(torch.uint8)  # image
        obs_emb, obs_feat = self.obs_enc((img.permute(0, 3, 1, 2), tac))
        joint_emb = torch.cat([state_emb, obs_emb], dim=1)

        actions_mean = self.actor(joint_emb)
        return actions_mean

    def evaluate(self, obs_features, state, actions):

        state_emb = self.state_enc(state)
        obs_emb = self.obs_enc.forward_feat(obs_features)
        joint_emb = torch.cat([state_emb, obs_emb], dim=1)
        actions_mean = self.actor(joint_emb)

        covariance = torch.diag(self.log_std.exp() * self.log_std.exp())
        distribution = MultivariateNormal(actions_mean, scale_tril=covariance)

        actions_log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()

        # obs_emb_vf = self.extract_feat(observations, self.extractors_vf)
        value = self.critic(joint_emb)

        return actions_log_prob, entropy, value, actions_mean, self.log_std.repeat(actions_mean.shape[0], 1)
class ActorCriticVEncoderT(nn.Module):

    def __init__(self, obs_shape, actions_shape, initial_std, model_cfg, encoder_cfg, env_cfg):
        super(ActorCriticVEncoderT, self).__init__()

        self.obs_dim = [v for v in env_cfg['obs_dim'].values()]
        # Encoder
        emb_dim = encoder_cfg["emb_dim"]

        self.obs_enc = EncoderVE_T(
            model_name=encoder_cfg["name"],
            pretrain_dir=encoder_cfg["pretrain_dir"],
            freeze=encoder_cfg["freeze"],
            emb_dim=emb_dim,
            en_mode=encoder_cfg["en_mode"]
        )

        self.state_enc = nn.Linear(env_cfg['obs_dim']['prop'], emb_dim)
        # self.obs_fc = nn.Linear(64, obs_emb)

        if model_cfg is None:
            actor_hidden_dim = [256, 256, 256]
            critic_hidden_dim = [256, 256, 256]
            activation = get_activation("selu")
        else:
            actor_hidden_dim = model_cfg['pi_hid_sizes']
            critic_hidden_dim = model_cfg['vf_hid_sizes']
            activation = get_activation(model_cfg['activation'])

        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(3*emb_dim, actor_hidden_dim[0]))
        actor_layers.append(activation)
        for l in range(len(actor_hidden_dim)):
            if l == len(actor_hidden_dim) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dim[l], *actions_shape))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dim[l], actor_hidden_dim[l + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)

        # Value function
        # self.extractors_vf = nn.ModuleList(LinearEncoder(self.obs_dim[i], hidden_size, obs_emb) for i in range(self.obs_dim.__len__()))
        critic_layers = []

        critic_layers.append(nn.Linear(3*emb_dim, critic_hidden_dim[0]))
        critic_layers.append(activation)
        for l in range(len(critic_hidden_dim)):
            if l == len(critic_hidden_dim) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dim[l], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dim[l], critic_hidden_dim[l + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        # print(self.obs_enc)
        # print(self.actor)
        # print(self.critic)

        # Action noise
        self.log_std = nn.Parameter(np.log(initial_std) * torch.ones(*actions_shape))

        # Initialize the weights like in stable baselines
        actor_weights = [np.sqrt(2)] * len(actor_hidden_dim)
        actor_weights.append(0.01)
        critic_weights = [np.sqrt(2)] * len(critic_hidden_dim)
        critic_weights.append(1.0)
        self.init_weights(self.actor, actor_weights)
        self.init_weights(self.critic, critic_weights)
    @staticmethod
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

    def obs_division(self, x):
        n_input_types = len(self.obs_dim)
        assert n_input_types > 1
        x_list = []
        st_idx = 0
        for idx in range(n_input_types):
            end_idx = st_idx + self.obs_dim[idx]
            x_list.append(x[:, st_idx:end_idx])
            st_idx += self.obs_dim[idx]

        return x_list
    # def extract_feat(self, x):
    #
    #     return o_en, state, obs_feat

    def forward(self):
        raise NotImplementedError

    @torch.no_grad()
    def act(self, observations):
        self.obs_enc.eval()
        state, img, tac = self.obs_division(observations)
        state_emb = self.state_enc(state)
        img = img.view(-1, 224, 224, 3).to(torch.uint8)  # image
        obs_emb, obs_feat = self.obs_enc((img.permute(0, 3, 1, 2), tac))
        joint_emb = torch.cat([state_emb, obs_emb], dim=1)


        actions_mean = self.actor(joint_emb)

        covariance = torch.diag(self.log_std.exp() * self.log_std.exp())
        distribution = MultivariateNormal(actions_mean, scale_tril=covariance)

        actions = distribution.sample()
        actions_log_prob = distribution.log_prob(actions)

        # obs_emb_vf = self.extract_feat(observations, self.extractors_vf)
        value = self.critic(joint_emb)
        self.obs_enc.train()

        return actions.detach(), \
               actions_log_prob.detach(), \
               value.detach(), \
               actions_mean.detach(), \
               self.log_std.repeat(actions_mean.shape[0], 1).detach(), \
               state.detach(),\
               obs_feat.detach()

    @torch.no_grad()
    def act_inference(self, observations):
        self.obs_enc.eval()
        state, img, tac = self.obs_division(observations)
        state_emb = self.state_enc(state)
        img = img.view(-1, 224, 224, 3).to(torch.uint8)  # image
        obs_emb, obs_feat = self.obs_enc((img.permute(0, 3, 1, 2), tac))
        joint_emb = torch.cat([state_emb, obs_emb], dim=1)

        actions_mean = self.actor(joint_emb)
        return actions_mean

    def evaluate(self, obs_features, state, actions):

        state_emb = self.state_enc(state)
        obs_emb = self.obs_enc.forward_feat(obs_features)
        joint_emb = torch.cat([state_emb, obs_emb], dim=1)
        actions_mean = self.actor(joint_emb)

        covariance = torch.diag(self.log_std.exp() * self.log_std.exp())
        distribution = MultivariateNormal(actions_mean, scale_tril=covariance)

        actions_log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()

        # obs_emb_vf = self.extract_feat(observations, self.extractors_vf)
        value = self.critic(joint_emb)

        return actions_log_prob, entropy, value, actions_mean, self.log_std.repeat(actions_mean.shape[0], 1)
class ActorCriticV(nn.Module):

    def __init__(self, obs_shape, actions_shape, initial_std, model_cfg, encoder_cfg, env_cfg):
        super(ActorCriticV, self).__init__()

        self.obs_dim = [v for v in env_cfg['obs_dim'].values()]
        # Encoder
        emb_dim = encoder_cfg["emb_dim"]

        self.obs_enc = EncoderV_no_pre(model_name=encoder_cfg["name"], emb_dim=emb_dim)
        self.state_enc = nn.Linear(env_cfg['obs_dim']['prop'], emb_dim)
        # self.obs_fc = nn.Linear(64, obs_emb)

        if model_cfg is None:
            actor_hidden_dim = [256, 256, 256]
            critic_hidden_dim = [256, 256, 256]
            activation = get_activation("selu")
        else:
            actor_hidden_dim = model_cfg['pi_hid_sizes']
            critic_hidden_dim = model_cfg['vf_hid_sizes']
            activation = get_activation(model_cfg['activation'])

        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(2*emb_dim, actor_hidden_dim[0]))
        actor_layers.append(activation)
        for l in range(len(actor_hidden_dim)):
            if l == len(actor_hidden_dim) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dim[l], *actions_shape))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dim[l], actor_hidden_dim[l + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)

        # Value function
        # self.extractors_vf = nn.ModuleList(LinearEncoder(self.obs_dim[i], hidden_size, obs_emb) for i in range(self.obs_dim.__len__()))
        critic_layers = []

        critic_layers.append(nn.Linear(2*emb_dim, critic_hidden_dim[0]))
        critic_layers.append(activation)
        for l in range(len(critic_hidden_dim)):
            if l == len(critic_hidden_dim) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dim[l], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dim[l], critic_hidden_dim[l + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        # print(self.obs_enc)
        # print(self.actor)
        # print(self.critic)

        # Action noise
        self.log_std = nn.Parameter(np.log(initial_std) * torch.ones(*actions_shape))

        # Initialize the weights like in stable baselines
        actor_weights = [np.sqrt(2)] * len(actor_hidden_dim)
        actor_weights.append(0.01)
        critic_weights = [np.sqrt(2)] * len(critic_hidden_dim)
        critic_weights.append(1.0)
        self.init_weights(self.actor, actor_weights)
        self.init_weights(self.critic, critic_weights)

    @staticmethod
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

    def obs_division(self, x):
        n_input_types = len(self.obs_dim)
        assert n_input_types > 1
        x_list = []
        st_idx = 0
        for idx in range(n_input_types):
            end_idx = st_idx + self.obs_dim[idx]
            x_list.append(x[:, st_idx:end_idx])
            st_idx += self.obs_dim[idx]

        return x_list
    # def extract_feat(self, x):
    #
    #     return o_en, state, obs_feat

    def forward(self):
        raise NotImplementedError

    @torch.no_grad()
    def act(self, observations):
        self.obs_enc.eval()
        state, img = self.obs_division(observations)
        state_emb = self.state_enc(state)
        img = img.view(-1, 224, 224, 3).to(torch.uint8)  # image
        obs_emb = self.obs_enc(img.permute(0, 3, 1, 2))
        joint_emb = torch.cat([state_emb, obs_emb], dim=1)


        actions_mean = self.actor(joint_emb)

        covariance = torch.diag(self.log_std.exp() * self.log_std.exp())
        distribution = MultivariateNormal(actions_mean, scale_tril=covariance)

        actions = distribution.sample()
        actions_log_prob = distribution.log_prob(actions)

        # obs_emb_vf = self.extract_feat(observations, self.extractors_vf)
        value = self.critic(joint_emb)
        self.obs_enc.train()

        return actions.detach(), \
               actions_log_prob.detach(), \
               value.detach(), \
               actions_mean.detach(), \
               self.log_std.repeat(actions_mean.shape[0], 1).detach(), \
               state.detach(),\
               img.detach()

    @torch.no_grad()
    def act_inference(self, observations):
        self.obs_enc.eval()
        state, img = self.obs_division(observations)
        state_emb = self.state_enc(state)
        img = img.view(-1, 224, 224, 3).to(torch.uint8)  # image
        obs_emb = self.obs_enc(img.permute(0, 3, 1, 2))
        joint_emb = torch.cat([state_emb, obs_emb], dim=1)

        actions_mean = self.actor(joint_emb)
        return actions_mean

    def evaluate(self, obs_features, state, actions):

        state_emb = self.state_enc(state)
        obs_emb = self.obs_enc(obs_features.permute(0, 3, 1, 2).to(torch.uint8))
        joint_emb = torch.cat([state_emb, obs_emb], dim=1)
        actions_mean = self.actor(joint_emb)

        covariance = torch.diag(self.log_std.exp() * self.log_std.exp())
        distribution = MultivariateNormal(actions_mean, scale_tril=covariance)

        actions_log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()

        # obs_emb_vf = self.extract_feat(observations, self.extractors_vf)
        value = self.critic(joint_emb)

        return actions_log_prob, entropy, value, actions_mean, self.log_std.repeat(actions_mean.shape[0], 1)
class ActorCriticT(nn.Module):

    def __init__(self, obs_shape, actions_shape, initial_std, model_cfg, encoder_cfg, env_cfg):
        super(ActorCriticT, self).__init__()

        self.obs_dim = [v for v in env_cfg['obs_dim'].values()]
        # Encoder
        emb_dim = encoder_cfg["emb_dim"]

        self.obs_enc = nn.Sequential(
                nn.Linear(20, 128),
                nn.ReLU(),
                nn.Linear(128, 256),
                nn.ReLU(),
                nn.Linear(256, emb_dim))
        self.state_enc = nn.Linear(env_cfg['obs_dim']['prop'], emb_dim)
        # self.obs_fc = nn.Linear(64, obs_emb)

        if model_cfg is None:
            actor_hidden_dim = [256, 256, 256]
            critic_hidden_dim = [256, 256, 256]
            activation = get_activation("selu")
        else:
            actor_hidden_dim = model_cfg['pi_hid_sizes']
            critic_hidden_dim = model_cfg['vf_hid_sizes']
            activation = get_activation(model_cfg['activation'])

        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(2*emb_dim, actor_hidden_dim[0]))
        actor_layers.append(activation)
        for l in range(len(actor_hidden_dim)):
            if l == len(actor_hidden_dim) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dim[l], *actions_shape))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dim[l], actor_hidden_dim[l + 1]))
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)

        # Value function
        # self.extractors_vf = nn.ModuleList(LinearEncoder(self.obs_dim[i], hidden_size, obs_emb) for i in range(self.obs_dim.__len__()))
        critic_layers = []

        critic_layers.append(nn.Linear(2*emb_dim, critic_hidden_dim[0]))
        critic_layers.append(activation)
        for l in range(len(critic_hidden_dim)):
            if l == len(critic_hidden_dim) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dim[l], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dim[l], critic_hidden_dim[l + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        # print(self.obs_enc)
        # print(self.actor)
        # print(self.critic)

        # Action noise
        self.log_std = nn.Parameter(np.log(initial_std) * torch.ones(*actions_shape))

        # Initialize the weights like in stable baselines
        actor_weights = [np.sqrt(2)] * len(actor_hidden_dim)
        actor_weights.append(0.01)
        critic_weights = [np.sqrt(2)] * len(critic_hidden_dim)
        critic_weights.append(1.0)
        self.init_weights(self.actor, actor_weights)
        self.init_weights(self.critic, critic_weights)

    @staticmethod
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

    def obs_division(self, x):
        n_input_types = len(self.obs_dim)
        assert n_input_types > 1
        x_list = []
        st_idx = 0
        for idx in range(n_input_types):
            end_idx = st_idx + self.obs_dim[idx]
            x_list.append(x[:, st_idx:end_idx])
            st_idx += self.obs_dim[idx]

        return x_list
    # def extract_feat(self, x):
    #
    #     return o_en, state, obs_feat

    def forward(self):
        raise NotImplementedError

    @torch.no_grad()
    def act(self, observations):
        self.obs_enc.eval()
        state, tac = self.obs_division(observations)
        state_emb = self.state_enc(state)
        # img = img.view(-1, 224, 224, 3).to(torch.uint8)  # image
        obs_emb = self.obs_enc(tac)
        joint_emb = torch.cat([state_emb, obs_emb], dim=1)


        actions_mean = self.actor(joint_emb)

        covariance = torch.diag(self.log_std.exp() * self.log_std.exp())
        distribution = MultivariateNormal(actions_mean, scale_tril=covariance)

        actions = distribution.sample()
        actions_log_prob = distribution.log_prob(actions)

        # obs_emb_vf = self.extract_feat(observations, self.extractors_vf)
        value = self.critic(joint_emb)
        self.obs_enc.train()

        return actions.detach(), \
               actions_log_prob.detach(), \
               value.detach(), \
               actions_mean.detach(), \
               self.log_std.repeat(actions_mean.shape[0], 1).detach(), \
               state.detach(),\
               tac.detach()

    @torch.no_grad()
    def act_inference(self, observations):
        self.obs_enc.eval()
        state, tac = self.obs_division(observations)
        state_emb = self.state_enc(state)
        # img = img.view(-1, 224, 224, 3).to(torch.uint8)  # image
        obs_emb = self.obs_enc(tac)
        joint_emb = torch.cat([state_emb, obs_emb], dim=1)

        actions_mean = self.actor(joint_emb)
        return actions_mean

    def evaluate(self, obs_features, state, actions):

        state_emb = self.state_enc(state)
        obs_emb = self.obs_enc(obs_features)
        joint_emb = torch.cat([state_emb, obs_emb], dim=1)
        actions_mean = self.actor(joint_emb)

        covariance = torch.diag(self.log_std.exp() * self.log_std.exp())
        distribution = MultivariateNormal(actions_mean, scale_tril=covariance)

        actions_log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()

        # obs_emb_vf = self.extract_feat(observations, self.extractors_vf)
        value = self.critic(joint_emb)

        return actions_log_prob, entropy, value, actions_mean, self.log_std.repeat(actions_mean.shape[0], 1)
def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None