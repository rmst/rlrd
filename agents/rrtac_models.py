from dataclasses import dataclass, InitVar

import gym
import torch
from agents.util import collate, partition
from agents.nn import TanhNormalLayer, SacLinear, big_conv
from torch.nn import Module, Linear, Sequential, ReLU, Conv2d, LeakyReLU, LSTM
from torch.nn.functional import leaky_relu


class ActorModule(Module):
  device = 'cpu'
  actor: callable
  actor_state = None

  # noinspection PyMethodOverriding
  def to(self, device):
    self.device = device
    return super().to(device=device)

  def act(self, obs, r, done, info, train=False):
    obs_col = collate((obs,), device=self.device)
    with torch.no_grad():
      action_distribution, self.actor_state = self.actor(obs_col, None if done else self.actor_state)
      action_col = action_distribution.sample() if train else action_distribution.sample_deterministic()
    action, = partition(action_col)
    return action, []


class LstmModel(ActorModule):
  def __init__(self, observation_space, action_space, hidden_units: int = 256):
    super().__init__()
    assert isinstance(observation_space, gym.spaces.Tuple)
    input_dim = sum(s.shape[0] for s in observation_space)
    self.lstm = LSTM(input_dim, hidden_units)
    self.hidden_units = hidden_units
    self.critic_layer = Linear(hidden_units, 1)
    self.actor_layer = TanhNormalLayer(hidden_units, action_space.shape[0])
    self.critic_output_layers = (self.critic_layer,)
    self.actor_state = None

  def actor(self, x, state):
    a, _, _, state = self(x, state)
    return a, state

  def forward(self, x, state):
    assert isinstance(x, tuple)
    x = torch.cat(x, dim=1)
    batchsize = x.shape[0]
    state = (torch.randn(1, batchsize, self.hidden_units), torch.randn(1, batchsize, self.hidden_units)) if state is None else state

    h, state = self.lstm(x, state)
    v = self.critic_layer(h)
    action_distribution = self.actor_layer(h)
    return action_distribution, (v,), (h,), state

