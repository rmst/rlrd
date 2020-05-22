# Delayed RTAC

from copy import deepcopy
from dataclasses import dataclass
from functools import reduce

import torch
from torch.nn.functional import mse_loss

import agents.sac
from agents.memory import TrajMemoryNoHidden
from agents.nn import no_grad, exponential_moving_average
from agents.util import partial

from agents.sac_models_rd import Mlp
from agents.envs import RandomDelayEnv
from collections import deque


@dataclass(eq=0)
class Agent(agents.sac.Agent):
    Model: type = agents.rtac_models.MlpDouble
    loss_alpha: float = 0.2

    def __post_init__(self, Env):
        with Env() as env:
            observation_space, action_space = env.observation_space, env.action_space
            self.sup_obs_delay = env.obs_delay_range.stop
            self.sup_act_delay = env.act_delay_range.stop
            print(f"DEBUG: self.sup_obs_delay: {self.sup_obs_delay}")
            print(f"DEBUG: self.sup_act_delay: {self.sup_act_delay}")
            self.act_buf_size = self.sup_obs_delay + self.sup_act_delay - 1  # - 1 because self.sup_act_delay is actually max_act_delay as defined in the paper (self.sup_obs_delay is max_obs_delay+1)
        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        model = self.Model(observation_space, action_space)
        self.model = model.to(device)
        self.model_target = no_grad(deepcopy(self.model))

        self.outputnorm = self.OutputNorm(self.model.critic_output_layers)
        self.outputnorm_target = self.OutputNorm(self.model_target.critic_output_layers)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.memory = TrajMemoryNoHidden(self.memory_size, self.batchsize, device, history=self.act_buf_size)
        self.traj_new_actions = [None, ] * self.act_buf_size
        self.traj_new_actions_log_prob = [None, ] * self.act_buf_size
        self.traj_new_augm_obs = [None, ] * self.act_buf_size

        self.is_training = False

    def train(self):
        # sample a trajectory of length self.act_buf_size
        augm_obs_traj, act_traj, rew_traj, terminals = self.memory.sample()

        batch_size = terminals.shape[0]
        print(f"DEBUG: batch_size: {batch_size}")
        print(f"DEBUG: augm_obs_traj: {augm_obs_traj}")
        print(f"DEBUG: act_traj: {act_traj}")
        print(f"DEBUG: rew_traj: {rew_traj}")
        print(f"DEBUG: terminals: {terminals}")

        # to determine the length of the n-step backup, nstep_len is the time at which the currently computed action (== i) or any action that followed (< i) has been applied first:
        int_tens_type = obs_del = augm_obs_traj[0][2].dtype
        ones_tens = torch.ones(batch_size, device=self.device, dtype=int_tens_type)
        nstep_len = ones_tens * self.act_buf_size
        for i in reversed(range(self.act_buf_size)):
            obs_del = augm_obs_traj[i][2]
            act_del = augm_obs_traj[i][3]
            tot_del = obs_del + act_del
            print(f"DEBUG: i: {i}")
            print(f"DEBUG: obs_del: {obs_del}")
            print(f"DEBUG: act_del: {act_del}")
            print(f"DEBUG: tot_del: {tot_del}")
            print(f"DEBUG: nstep_len before: {nstep_len}")
            nstep_len = torch.where((tot_del <= i) & (tot_del < nstep_len), ones_tens * i, nstep_len)  # FIXME: check that this works as expected
            print(f"DEBUG: nstep_len after: {nstep_len}")
        print(f"DEBUG:nstep_len: {nstep_len}")
        nstep_max_len = torch.max(nstep_len)
        assert nstep_max_len < self.act_buf_size, "Delays longer than the action buffer (e.g. infinite) are not supported"
        print(f"DEBUG:nstep_max_len: {nstep_max_len}")
        nstep_one_hot = torch.zeros(len(nstep_len), nstep_max_len + 1).scatter_(1, nstep_len.unsqueeze(1), 1.)
        print(f"DEBUG:nstep_one_hot: {nstep_one_hot}")

        # use the current policy to compute a new trajectory of actions of length self.act_buf_size
        for i in range(self.act_buf_size):
            # compute a new action and update the corresponding augmented observation:
            augm_obs = augm_obs_traj[i]
            print(f"DEBUG: augm_obs before: {augm_obs}")
            if i > 0:
                # FIXME: check that this won't mess with autograd
                act_slice = tuple(self.traj_new_actions[self.act_buf_size - i:self.act_buf_size])  # FIXME: check that first action in the action buffer is indeed the last computed action
                augm_obs = augm_obs[:1] + ((act_slice + augm_obs[1][i:]), ) + augm_obs[2:]
                print(f"DEBUG: augm_obs after: {augm_obs}")
            new_action_distribution = self.model.actor(augm_obs)
            # this is stored in right -> left order for replacing correctly in augm_obs:
            self.traj_new_actions[self.act_buf_size - i - 1] = new_action_distribution.rsample()
            # this is stored in right -> left order for consistency:
            self.traj_new_actions_log_prob[self.act_buf_size - i - 1] = new_action_distribution.log_prob(self.traj_new_actions[self.act_buf_size - i - 1])
        print(f"DEBUG: self.traj_new_actions: {self.traj_new_actions}")
        print(f"DEBUG: self.traj_new_actions_log_prob: {self.traj_new_actions_log_prob}")

        assert False

        # we now compute the state-value target:
        # val contains the state-value estimates at step t+1+nstep_len: v() (can be different for each element of the batch)
        for i in reversed(range(nstep_max_len)):
            rew = rew_traj[i]
            val = rew + (1. - nstep_one_hot) * self.discount * val

        next_val_ = [c(obs_, act_) for c in self.model_target.critics]
        next_val_ = reduce(torch.min, torch.stack(next_val_))
        # XXX not normalizing

        # accumulate
        val_ = next_val_[:, :, 0]  # XXX ignore entropy component for now
        for rew_, don_ in zip(reversed(rewards_traj), reversed(done_mask)):
            val_ = rew_ + (1. - don_.float()) * self.discount * val_


        # critic loss
        _, next_value_target, _ = self.model_target((next_obs[0], new_actions.detach()))
        next_value_target = reduce(torch.min, next_value_target)
        next_value_target = self.outputnorm_target.unnormalize(next_value_target)

        reward_components = torch.stack((
            self.reward_scale * rewards,
            - self.entropy_scale * new_actions_log_prob.detach(),
        ), dim=1)

        value_target = reward_components + (1. - terminals[:, None]) * self.discount * next_value_target
        # TODO: is it really that helpful/necessary to do the outnorm update here and to recompute the values?
        value_target = self.outputnorm.update(value_target)
        values = tuple(c(h) for c, h in zip(self.model.critic_output_layers, hidden))  # recompute values (weights changed)

        assert values[0].shape == value_target.shape and not value_target.requires_grad
        loss_critic = sum(mse_loss(v, value_target) for v in values)

        # actor loss
        _, next_value, _ = self.model_nograd((next_obs[0], new_actions))
        next_value = reduce(torch.min, next_value)
        new_value = (1. - terminals[:, None]) * self.discount * self.outputnorm.unnormalize(next_value)
        new_value[:, -1] -= self.entropy_scale * new_actions_log_prob
        assert new_value.shape == (self.batchsize, 2)
        loss_actor = - self.outputnorm.normalize_sum(new_value.sum(1)).mean()  # normalize_sum preserves relative scale

        # update model
        self.optimizer.zero_grad()
        loss_total = self.loss_alpha * loss_actor + (1 - self.loss_alpha) * loss_critic
        loss_total.backward()
        self.optimizer.step()

        # update target model and normalizers
        exponential_moving_average(self.model_target.parameters(), self.model.parameters(), self.target_update)
        exponential_moving_average(self.outputnorm_target.parameters(), self.outputnorm.parameters(), self.target_update)

        return dict(
            loss_total=loss_total.detach(),
            loss_critic=loss_critic.detach(),
            loss_actor=loss_actor.detach(),
            outputnorm_reward_mean=self.outputnorm.mean[0],
            outputnorm_entropy_mean=self.outputnorm.mean[-1],
            outputnorm_reward_std=self.outputnorm.std[0],
            outputnorm_entropy_std=self.outputnorm.std[-1],
            memory_size=len(self.memory),
            # entropy_scale=self.entropy_scale
        )


if __name__ == "__main__":
    from agents import Training, run

    DacTest = partial(
        Training,
        epochs=3,
        rounds=5,
        steps=500,
        Agent=partial(Agent, device='cpu', memory_size=1000000, start_training=256, batchsize=4, Model=Mlp),
        Env=partial(RandomDelayEnv, min_observation_delay=1, sup_observation_delay=3, min_action_delay=1, sup_action_delay=2),
    )

    run(DacTest)
# run(Rtac_Avenue_Test)
