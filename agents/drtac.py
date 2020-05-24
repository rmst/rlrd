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

from agents.drtac_models import Mlp
from agents.envs import RandomDelayEnv


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
        self.traj_new_augm_obs = [None, ] * (self.act_buf_size + 1)  # + 1 because the trajectory is obs0 -> rew1(obs0,act0) -> obs1 -> ...

        self.is_training = False

    def train(self):
        # TODO: remove requires_grad everywhere it should not be

        # sample a trajectory of length self.act_buf_size
        # NB: when terminals is True, the terminal (delayed) augmented state is the last one of the trajectory
        augm_obs_traj, act_traj, rew_traj, terminals = self.memory.sample()

        batch_size = terminals.shape[0]
        print(f"DEBUG: batch_size: {batch_size}")
        print(f"DEBUG: augm_obs_traj: {augm_obs_traj}")
        print(f"DEBUG: act_traj: {act_traj}")
        print(f"DEBUG: rew_traj: {rew_traj}")
        print(f"DEBUG: terminals: {terminals}")

        # value of the first augmented state:
        values = [c(augm_obs_traj[0]).squeeze() for c in self.model.critics]
        print(f"DEBUG: values: {values}")

        # to determine the length of the n-step backup, nstep_len is the time at which the currently computed action (== i) or any action that followed (< i) has been applied first:
        # when nstep_len is k (in 0..self.act_buf_size-1), it means that the action computed with the first augmented observation of the trajectory will have an effect k+1 steps later
        # (either it will be applied, or an action that follows it will be applied)
        int_tens_type = obs_del = augm_obs_traj[0][2].dtype
        ones_tens = torch.ones(batch_size, device=self.device, dtype=int_tens_type)
        nstep_len = ones_tens * self.act_buf_size
        for i in reversed(range(self.act_buf_size)):  # caution: we don't care about the delay of the first observation in the trajectory, but we care about the last one
            obs_del = augm_obs_traj[i + 1][2]
            act_del = augm_obs_traj[i + 1][3]
            tot_del = obs_del + act_del
            print(f"DEBUG: i + 1: {i + 1}")
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
        for i in range(self.act_buf_size + 1):
            # compute a new action and update the corresponding *next* augmented observation:
            augm_obs = augm_obs_traj[i]  # FIXME: this modifies augm_obs_traj, check that this is not an issue
            print(f"DEBUG: augm_obs at index {i}: {augm_obs}")
            if i > 0:
                # FIXME: check that this won't mess with autograd
                act_slice = tuple(self.traj_new_actions[self.act_buf_size - i:self.act_buf_size])  # FIXME: check that first action in the action buffer is indeed the last computed action
                augm_obs = augm_obs[:1] + ((act_slice + augm_obs[1][i:]), ) + augm_obs[2:]
                print(f"DEBUG: augm_obs at index {i} after replacing actions: {augm_obs}")
            if i < self.act_buf_size:  # we don't compute the action for the last observation of the trajectory
                new_action_distribution = self.model.actor(augm_obs)
                # this is stored in right -> left order for replacing correctly in augm_obs:
                self.traj_new_actions[self.act_buf_size - i - 1] = new_action_distribution.rsample()
                # this is stored in left -> right order for to be consistent with the reward trajectory:
                self.traj_new_actions_log_prob[i] = new_action_distribution.log_prob(self.traj_new_actions[self.act_buf_size - i - 1])
            # this is stored in left -> right order:
            self.traj_new_augm_obs[i] = augm_obs
        print(f"DEBUG: self.traj_new_actions: {self.traj_new_actions}")
        print(f"DEBUG: self.traj_new_actions_log_prob: {self.traj_new_actions_log_prob}")
        print(f"DEBUG: self.traj_new_augm_obs: {self.traj_new_augm_obs}")

        # We now compute the state-value estimate of the augmented states at which the computed actions will be applied for each trajectory of the batch
        # (caution: this can be a different position in the trajectory for each element of the batch).

        # We expect each augmented state to be of shape (obs:tensor, act_buf:(tensor, ..., tensor), obs_del:tensor, act_del:tensor). Each tensor is batched.
        # We want to execute only 1 forward pass in the state-value estimator, therefore we recreate an artificially batched augmented state for this specific purpose.

        print(f"DEBUG: nstep_len: {nstep_len}")
        obs_s = torch.stack([self.traj_new_augm_obs[i + 1][0][ibatch] for ibatch, i in enumerate(nstep_len)])
        act_s = tuple(torch.stack([self.traj_new_augm_obs[i + 1][1][iact][ibatch] for ibatch, i in enumerate(nstep_len)]) for iact in range(self.act_buf_size))
        od_s = torch.stack([self.traj_new_augm_obs[i + 1][2][ibatch] for ibatch, i in enumerate(nstep_len)])
        ad_s = torch.stack([self.traj_new_augm_obs[i + 1][3][ibatch] for ibatch, i in enumerate(nstep_len)])
        mod_augm_obs = tuple((obs_s, act_s, od_s, ad_s))
        print(f"DEBUG: mod_augm_obs: {mod_augm_obs}")

        print("DEBUG: --- CRITIC LOSS ---")

        with torch.no_grad():

            # These are the delayed state-value estimates we are looking for:
            mod_val = [c(mod_augm_obs) for c in self.model_target.critics]
            mod_val = reduce(torch.min, torch.stack(mod_val)).squeeze()  # minimum target estimate
            print(f"DEBUG: mod_val before removing terminal states: {mod_val}")
            mod_val = mod_val * (1. - terminals)
            print(f"DEBUG: mod_val after removing terminal states: {mod_val}")

            # Now let us use this to compute the state-value targets of the batch of initial augmented states:

            value_target = torch.zeros(batch_size)
            backup_started = torch.zeros(batch_size)
            print(f"DEBUG: self.discount: {self.discount}")
            print(f"DEBUG: self.reward_scale: {self.reward_scale}")
            print(f"DEBUG: self.entropy_scale: {self.entropy_scale}")
            print(f"DEBUG: terminals: {terminals}")
            for i in reversed(range(nstep_max_len + 1)):
                start_backup_mask = nstep_one_hot[:, i]
                backup_started += start_backup_mask
                print(f"DEBUG: i: {i}")
                print(f"DEBUG: start_backup_mask: {start_backup_mask}")
                print(f"DEBUG: backup_started: {backup_started}")
                value_target = self.reward_scale * rew_traj[i] - self.entropy_scale * self.traj_new_actions_log_prob[i] + backup_started * self.discount * (value_target + start_backup_mask * mod_val)
                print(f"DEBUG: rew_traj[i]: {rew_traj[i]}")
                print(f"DEBUG: self.traj_new_actions_log_prob[i]: {self.traj_new_actions_log_prob[i]}")
                print(f"DEBUG: new value_target: {value_target}")
            print(f"DEBUG: state-value target: {value_target}")

        # end of torch.no_grad()

        assert values[0].shape == value_target.shape, f"values[0].shape : {values[0].shape} != value_target.shape : {value_target.shape}"
        assert not value_target.requires_grad

        # Now the critic loss is:

        loss_critic = sum(mse_loss(v, value_target) for v in values)
        print(f"DEBUG: loss_critic: {loss_critic}")

        # actor loss:
        # TODO: there is probably a way of merging this with the previous for loop

        print("DEBUG: --- ACTOR LOSS ---")
        model_mod_val = [c(mod_augm_obs) for c in self.model_nograd.critics]
        model_mod_val = reduce(torch.min, torch.stack(model_mod_val)).squeeze()  # minimum model estimate
        print(f"DEBUG: model_mod_val: {model_mod_val}")
        loss_actor = torch.zeros(batch_size)
        backup_started = torch.zeros(batch_size)
        print(f"DEBUG: self.discount: {self.discount}")
        print(f"DEBUG: self.reward_scale: {self.reward_scale}")
        print(f"DEBUG: self.entropy_scale: {self.entropy_scale}")
        print(f"DEBUG: terminals: {terminals}")
        for i in reversed(range(nstep_max_len + 1)):
            start_backup_mask = nstep_one_hot[:, i]
            backup_started += start_backup_mask
            print(f"DEBUG: i: {i}")
            print(f"DEBUG: start_backup_mask: {start_backup_mask}")
            print(f"DEBUG: backup_started: {backup_started}")
            # TODO: check signs and indexes (WIP)
            loss_actor = - self.entropy_scale * self.traj_new_actions_log_prob[i] + backup_started * self.discount * (loss_actor + start_backup_mask * model_mod_val)
            print(f"DEBUG: self.traj_new_actions_log_prob[i]: {self.traj_new_actions_log_prob[i]}")
            print(f"DEBUG: new loss_actor: {loss_actor}")
        loss_actor = loss_actor * -1.0
        loss_actor = loss_actor.sum()
        print(f"DEBUG: final loss_actor: {loss_actor}")

        assert False

        # update model
        self.optimizer.zero_grad()
        loss_total = self.loss_alpha * loss_actor + (1 - self.loss_alpha) * loss_critic
        loss_total.backward()
        self.optimizer.step()

        # update target model and normalizers
        exponential_moving_average(self.model_target.parameters(), self.model.parameters(), self.target_update)
        # exponential_moving_average(self.outputnorm_target.parameters(), self.outputnorm.parameters(), self.target_update)

        return dict(
            loss_total=loss_total.detach(),
            loss_critic=loss_critic.detach(),
            loss_actor=loss_actor.detach(),
            # outputnorm_reward_mean=self.outputnorm.mean[0],
            # outputnorm_entropy_mean=self.outputnorm.mean[-1],
            # outputnorm_reward_std=self.outputnorm.std[0],
            # outputnorm_entropy_std=self.outputnorm.std[-1],
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
        Env=partial(RandomDelayEnv, min_observation_delay=0, sup_observation_delay=2, min_action_delay=0, sup_action_delay=2),
    )

    run(DacTest)
# run(Rtac_Avenue_Test)
