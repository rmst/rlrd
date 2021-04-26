# Delay Correcting Actor-Critic

from copy import deepcopy
from dataclasses import dataclass
from functools import reduce
import torch
from torch.nn.functional import mse_loss
import rlrd.sac
from rlrd.memory import TrajMemoryNoHidden
from rlrd.nn import no_grad, exponential_moving_average
from rlrd.util import partial
from rlrd.dcac_models import Mlp
from rlrd.envs import RandomDelayEnv
from rlrd import Training


@dataclass(eq=0)
class Agent(rlrd.sac.Agent):
    Model: type = Mlp
    loss_alpha: float = 0.2
    rtac: bool = False

    def __post_init__(self, Env):
        with Env() as env:
            observation_space, action_space = env.observation_space, env.action_space
            self.sup_obs_delay = env.obs_delay_range.stop
            self.sup_act_delay = env.act_delay_range.stop
            self.act_buf_size = self.sup_obs_delay + self.sup_act_delay - 1
            self.old_act_buf_size = deepcopy(self.act_buf_size)
            if self.rtac:
                self.act_buf_size = 1

        assert self.device is not None
        device = self.device  # or ("cuda" if torch.cuda.is_available() else "cpu")
        model = self.Model(observation_space, action_space)
        self.model = model.to(device)
        self.model_target = no_grad(deepcopy(self.model))

        self.outputnorm = self.OutputNorm(self.model.critic_output_layers)
        self.outputnorm_target = self.OutputNorm(self.model_target.critic_output_layers)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.memory = TrajMemoryNoHidden(self.memory_size, self.batchsize, device, history=self.act_buf_size)
        self.traj_new_actions = [None, ] * self.act_buf_size
        self.traj_new_actions_detach = [None, ] * self.act_buf_size
        self.traj_new_actions_log_prob = [None, ] * self.act_buf_size
        self.traj_new_actions_log_prob_detach = [None, ] * self.act_buf_size
        self.traj_new_augm_obs = [None, ] * (self.act_buf_size + 1)

        self.is_training = False

    def train(self):
        # sample a trajectory of length self.act_buf_size
        # NB: when terminals is True, the terminal augmented state is the last one of the trajectory (this is ensured by the sampling procedure)

        # TODO: act_traj is useless, it could be removed from the replay memory
        # FIXME: the profiler indicates that memory is inefficient, optimize

        augm_obs_traj, act_traj, rew_traj, terminals = self.memory.sample()

        batch_size = terminals.shape[0]

        # value of the first augmented state:
        values = [c(augm_obs_traj[0]).squeeze() for c in self.model.critics]

        # nstep_len is the number of valid transitions of the sampled sub-trajectory, not counting the first one which is always valid since we consider the action delay to be always >= 1.
        # nstep_len will be e.g. 0 in the rtrl setting (an action delay of 0 here means an action delay of 1 in the paper).

        int_tens_type = obs_del = augm_obs_traj[0][2].dtype
        ones_tens = torch.ones(batch_size, device=self.device, dtype=int_tens_type, requires_grad=False)

        if not self.rtac:
            nstep_len = ones_tens * (self.act_buf_size - 1)
            for i in reversed(range(self.act_buf_size)):  # we don't care about the delay of the first observation in the trajectory, but we care about the last one
                obs_del = augm_obs_traj[i + 1][2]  # observation delay (alpha)
                act_del = augm_obs_traj[i + 1][4]  # action_delay (beta)
                tot_del = obs_del + act_del
                # TODO: the last iteration is useless
                nstep_len = torch.where((tot_del <= i), ones_tens * (i - 1), nstep_len)
            nstep_max_len = torch.max(nstep_len)
            nstep_min_len = torch.min(nstep_len)
            assert nstep_min_len >= 0, "Each total delay must be at least 1 (instantaneous turn-based RL not supported)"
            nstep_one_hot = torch.zeros(len(nstep_len), nstep_max_len + 1, device=self.device, requires_grad=False).scatter_(1, nstep_len.unsqueeze(1).long(), 1.)
        else:  # RTAC is equivalent to doing only 1-step backups (i.e. nstep_len==0)
            nstep_len = torch.zeros(batch_size, device=self.device, dtype=int_tens_type, requires_grad=False)
            nstep_max_len = torch.max(nstep_len)
            nstep_one_hot = torch.zeros(len(nstep_len), nstep_max_len + 1, device=self.device, requires_grad=False).scatter_(1, nstep_len.unsqueeze(1).long(), 1.)
            terminals = terminals if self.act_buf_size == 1 else terminals * 0.0  # the way the replay memory works, RTAC will never encounter terminal states for buffers of more than 1 action

        # use the current policy to compute a new trajectory of actions of length self.act_buf_size
        for i in range(self.act_buf_size + 1):
            # compute a new action and update the corresponding *next* augmented observation:
            augm_obs = augm_obs_traj[i]
            if i > 0:
                act_slice = tuple(self.traj_new_actions[self.act_buf_size - i:self.act_buf_size])
                augm_obs = augm_obs[:1] + ((act_slice + augm_obs[1][i:]), ) + augm_obs[2:]
            if i < self.act_buf_size:  # we don't compute the action for the last observation of the trajectory
                new_action_distribution = self.model.actor(augm_obs)
                # this is stored in right -> left order for replacing correctly in augm_obs:
                self.traj_new_actions[self.act_buf_size - i - 1] = new_action_distribution.rsample()
                self.traj_new_actions_detach[self.act_buf_size - i - 1] = self.traj_new_actions[self.act_buf_size - i - 1].detach()
                # this is stored in left -> right order for to be consistent with the reward trajectory:
                self.traj_new_actions_log_prob[i] = new_action_distribution.log_prob(self.traj_new_actions[self.act_buf_size - i - 1])
                self.traj_new_actions_log_prob_detach[i] = self.traj_new_actions_log_prob[i].detach()
            # this is stored in left -> right order:
            self.traj_new_augm_obs[i] = augm_obs

        # We now compute the state-value estimate
        # (this can be a different position in the trajectory for each element of the batch).
        # We expect each augmented state to be of shape (obs:tensor, act_buf:(tensor, ..., tensor), obs_del:tensor, act_del:tensor). Each tensor is batched.
        # To execute only 1 forward pass in the state-value estimator we recreate an artificially batched augmented state for this specific purpose.

        # FIXME: the profiler indicates that the following 5 lines are very inefficient, optimize

        obs_s = torch.stack([self.traj_new_augm_obs[i + 1][0][ibatch] for ibatch, i in enumerate(nstep_len)])
        act_s = tuple(torch.stack([self.traj_new_augm_obs[i + 1][1][iact][ibatch] for ibatch, i in enumerate(nstep_len)]) for iact in range(self.old_act_buf_size))
        od_s = torch.stack([self.traj_new_augm_obs[i + 1][2][ibatch] for ibatch, i in enumerate(nstep_len)])
        ad_s = torch.stack([self.traj_new_augm_obs[i + 1][3][ibatch] for ibatch, i in enumerate(nstep_len)])
        mod_augm_obs = tuple((obs_s, act_s, od_s, ad_s))

        with torch.no_grad():

            # These are the delayed state-value estimates we are looking for:
            target_mod_val = [c(mod_augm_obs) for c in self.model_target.critics]
            target_mod_val = reduce(torch.min, torch.stack(target_mod_val)).squeeze()  # minimum target estimate
            target_mod_val = target_mod_val * (1. - terminals)

            # Now let us use this to compute the state-value targets of the batch of initial augmented states:

            value_target = torch.zeros(batch_size, device=self.device)
            backup_started = torch.zeros(batch_size, device=self.device)
            for i in reversed(range(nstep_max_len + 1)):
                start_backup_mask = nstep_one_hot[:, i]
                backup_started += start_backup_mask
                value_target = self.reward_scale * rew_traj[i] - self.entropy_scale * self.traj_new_actions_log_prob_detach[i] + backup_started * self.discount * (value_target + start_backup_mask * target_mod_val)

        assert values[0].shape == value_target.shape, f"values[0].shape : {values[0].shape} != value_target.shape : {value_target.shape}"
        assert not value_target.requires_grad

        # Now the critic loss is:

        loss_critic = sum(mse_loss(v, value_target) for v in values)

        # actor loss:
        # TODO: there is probably a way of merging this with the previous for loop

        model_mod_val = [c(mod_augm_obs) for c in self.model_nograd.critics]
        model_mod_val = reduce(torch.min, torch.stack(model_mod_val)).squeeze()  # minimum model estimate
        model_mod_val = model_mod_val * (1. - terminals)

        loss_actor = torch.zeros(batch_size, device=self.device)
        backup_started = torch.zeros(batch_size, device=self.device)
        for i in reversed(range(nstep_max_len + 1)):
            start_backup_mask = nstep_one_hot[:, i]
            backup_started += start_backup_mask
            loss_actor = - self.entropy_scale * self.traj_new_actions_log_prob[i] + backup_started * self.discount * (loss_actor + start_backup_mask * model_mod_val)
        loss_actor = - loss_actor.mean(0)

        # update model
        self.optimizer.zero_grad()
        loss_total = self.loss_alpha * loss_actor + (1 - self.loss_alpha) * loss_critic
        loss_total.backward()
        self.optimizer.step()

        # update target model
        exponential_moving_average(self.model_target.parameters(), self.model.parameters(), self.target_update)

        # exponential_moving_average(self.outputnorm_target.parameters(), self.outputnorm.parameters(), self.target_update)  # this is for trying PopArt in the future

        return dict(
            loss_total=loss_total.detach(),
            loss_critic=loss_critic.detach(),
            loss_actor=loss_actor.detach(),
            memory_size=len(self.memory),
        )
