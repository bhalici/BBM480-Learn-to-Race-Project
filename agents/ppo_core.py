import numpy as np
import scipy.signal
from gym.spaces import Box, Discrete

import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def mlp(sizes, activation=nn.ReLU, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.
    input: 
        vector x, 
        [x0, 
         x1, 
         x2]
    output:
        [x0 + discount * x1 + discount^2 * x2,  
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class Actor(nn.Module):

    def _distribution(self, obs):
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError

    def forward(self, obs, act=None):
        # Produce action distributions for given observations, and 
        # optionally compute the log likelihood of given actions under
        # those distributions.
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a
    


class MLPCategoricalActor(Actor):
    
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.logits_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        logits = self.logits_net(obs)
        return Categorical(logits=logits)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)


class MLPGaussianActor(Actor):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self.mu_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        mu = self.mu_net(obs)
        std = torch.exp(self.log_std)
        return Normal(mu, std)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(axis=-1)    # Last axis sum needed for Torch Normal distribution
    

class MLPGaussianActor_l2r(nn.Module):

    def __init__(self, cfg, activation=nn.ReLU):
        super().__init__()
        self.cfg = cfg

        log_std = -0.5 * np.ones(2, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))

        # pdb.set_trace()
        self.speed_encoder = mlp(
            [1] + self.cfg[self.cfg["use_encoder_type"]]["speed_hiddens"]
        )
        self.mu_net = mlp(
            [
                self.cfg[self.cfg["use_encoder_type"]]["latent_dims"]
                + self.cfg[self.cfg["use_encoder_type"]]["speed_hiddens"][-1]
            ]
            + self.cfg[self.cfg["use_encoder_type"]]["hiddens"]
            + [2]
            , activation=activation
        )

    def _distribution(self, obs_feat):

        img_embed = obs_feat[
            ..., : self.cfg[self.cfg["use_encoder_type"]]["latent_dims"]
        ]  # n x latent_dims
        speed = obs_feat[
            ..., self.cfg[self.cfg["use_encoder_type"]]["latent_dims"] :
        ]  # n x 1
        spd_embed = self.speed_encoder(speed)  # n x 16
        mu = self.mu_net(torch.cat([img_embed, spd_embed], dim=-1))  # n x 1

        std = torch.exp(self.log_std)
        return Normal(mu, std)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(axis=-1)    # Last axis sum needed for Torch Normal distribution
    
    def forward(self, obs_feat, act=None):
        # Produce action distributions for given observations, and 
        # optionally compute the log likelihood of given actions under
        # those distributions.
        pi = self._distribution(obs_feat)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a
    
    

class MLPCritic(nn.Module):

    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs):
        return torch.squeeze(self.v_net(obs), -1) # Critical to ensure v has right shape.

class Vfunction(nn.Module):

    def __init__(self, cfg, activation=nn.ReLU):
        super().__init__()
        self.cfg = cfg
        # pdb.set_trace()
        self.speed_encoder = mlp(
            [1] + self.cfg[self.cfg["use_encoder_type"]]["speed_hiddens"]
        )
        self.regressor = mlp(
            [
                self.cfg[self.cfg["use_encoder_type"]]["latent_dims"]
                + self.cfg[self.cfg["use_encoder_type"]]["speed_hiddens"][-1]
            ]
            + self.cfg[self.cfg["use_encoder_type"]]["hiddens"]
            + [1]
            , activation=activation
        )
        # self.lr = cfg['resnet']['LR']

    def forward(self, obs_feat):
        # if obs_feat.ndimension() == 1:
        #    obs_feat = obs_feat.unsqueeze(0)
        img_embed = obs_feat[
            ..., : self.cfg[self.cfg["use_encoder_type"]]["latent_dims"]
        ]  # n x latent_dims
        speed = obs_feat[
            ..., self.cfg[self.cfg["use_encoder_type"]]["latent_dims"] :
        ]  # n x 1
        spd_embed = self.speed_encoder(speed)  # n x 16
        out = self.regressor(torch.cat([img_embed, spd_embed], dim=-1))  # n x 1
        # pdb.set_trace()
        return out.view(-1)


class MLPActorCritic(nn.Module):


    def __init__(self, observation_space, action_space, 
                 hidden_sizes=(64,64), activation=nn.Tanh):
        super().__init__()

        obs_dim = observation_space.shape[0]

        self.pi = MLPGaussianActor(obs_dim, action_space.shape[0], hidden_sizes, activation)

        # build value function
        self.v  = MLPCritic(obs_dim, hidden_sizes, activation)

    def step(self, obs):
        with torch.no_grad():
            pi = self.pi._distribution(obs)
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v(obs)
        return a.numpy(), v.numpy(), logp_a.numpy()

    def act(self, obs):
        return self.step(obs)[0]
    

class ActorCritic(nn.Module):


    def __init__(self,
                 cfg,
                 activation=nn.ReLU,
                 device="cpu"):
        super().__init__()


        self.pi = MLPGaussianActor_l2r(cfg, activation)

        # build value function
        self.v  = Vfunction(cfg, activation)
        
        self.device = device
        self.to(self.device)

    def step(self, obs_feat):
        with torch.no_grad():
            pi = self.pi._distribution(obs_feat)
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v(obs_feat)
            
        if self.device == "cpu":
            a_np = a.numpy()
            v_np = v.numpy()
            logp_a_np = logp_a.numpy()
        
        else:
            a_np = a.cpu().numpy()
            v_np = v.cpu().numpy()
            logp_a_np = logp_a.cpu().numpy()

        return a_np, v_np, logp_a_np

    def act(self, obs_feat):
        return self.step(obs_feat)[0]
