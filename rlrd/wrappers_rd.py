from collections import deque
from random import sample
import itertools

import gym
from gym.spaces import Tuple, Discrete

import numpy as np


class RandomDelayWrapper(gym.Wrapper):
    """
    Wrapper for any non-RTRL environment, modelling random observation and action delays
    NB: alpha refers to the abservation delay, it is >= 0
    NB: The state-space now contains two different action delays:
        kappa is such that alpha+kappa is the index of the first action that was going to be applied when the observation started being captured, it is useful for the model
            (when kappa==0, it means that the delay is actually 1)
        beta is such that alpha+beta is the index of the last action that is known to have influenced the observation, it is useful for credit assignment (e.g. AC/DC)
            (alpha+beta is often 1 step bigger than the action buffer, and it is always >= 1)
    Kwargs:
        obs_delay_range: range in which alpha is sampled
        act_delay_range: range in which kappa is sampled
        initial_action: action (default None): action with which the action buffer is filled at reset() (if None, sampled in the action space)
    """

    def __init__(self, env, obs_delay_range=range(0, 8), act_delay_range=range(0, 2), initial_action=None, skip_initial_actions=False):
        super().__init__(env)
        self.wrapped_env = env
        self.obs_delay_range = obs_delay_range
        self.act_delay_range = act_delay_range

        self.observation_space = Tuple((
            env.observation_space,  # most recent observation
            Tuple([env.action_space] * (obs_delay_range.stop + act_delay_range.stop - 1)),  # action buffer
            Discrete(obs_delay_range.stop),  # observation delay int64
            Discrete(act_delay_range.stop),  # action delay int64
        ))

        self.initial_action = initial_action
        self.skip_initial_actions = skip_initial_actions
        self.past_actions = deque(maxlen=obs_delay_range.stop + act_delay_range.stop)
        self.past_observations = deque(maxlen=obs_delay_range.stop)
        self.arrival_times_actions = deque(maxlen=act_delay_range.stop)
        self.arrival_times_observations = deque(maxlen=obs_delay_range.stop)

        self.t = 0
        self.done_signal_sent = False
        self.next_action = None
        self.cum_rew_actor = 0.
        self.cum_rew_brain = 0.
        self.prev_action_idx = 0  # TODO : initialize this better

    def reset(self, **kwargs):
        self.cum_rew_actor = 0.
        self.cum_rew_brain = 0.
        self.prev_action_idx = 0  # TODO : initialize this better
        self.done_signal_sent = False
        first_observation = super().reset(**kwargs)

        # fill up buffers
        self.t = - (self.obs_delay_range.stop + self.act_delay_range.stop)  # this is <= -2
        while self.t < 0:
            act = self.action_space.sample() if self.initial_action is None else self.initial_action
            self.send_action(act, init=True)  # TODO : initialize this better
            self.send_observation((first_observation, 0., False, {}, 0, 1))  # TODO : initialize this better
            self.t += 1
        self.receive_action()  # an action has to be applied

        assert self.t == 0
        received_observation, *_ = self.receive_observation()
        # print("DEBUG: end of reset ---")
        # print(f"DEBUG: self.past_actions:{self.past_actions}")
        # print(f"DEBUG: self.past_observations:{self.past_observations}")
        # print(f"DEBUG: self.arrival_times_actions:{self.arrival_times_actions}")
        # print(f"DEBUG: self.arrival_times_observations:{self.arrival_times_observations}")
        # print(f"DEBUG: self.t:{self.t}")
        # print("DEBUG: ---")
        return received_observation

    def step(self, action):
        """
        When kappa is 0 and alpha is 0, this is equivalent to the RTRL setting
        (The inference time is NOT considered part of beta or kappa)
        """

        # at the brain
        self.send_action(action)

        # at the remote actor
        if self.t < self.act_delay_range.stop and self.skip_initial_actions:
            # assert False, "skip_initial_actions==True is not supported"
            # do nothing until the brain's first actions arrive at the remote actor
            self.receive_action()
        elif self.done_signal_sent:
            # just resend the last observation until the brain gets it
            self.send_observation(self.past_observations[0])
        else:
            m, r, d, info = self.env.step(self.next_action)  # before receive_action (e.g. rtrl setting with 0 delays)
            kappa, beta = self.receive_action()
            self.cum_rew_actor += r
            self.done_signal_sent = d
            self.send_observation((m, self.cum_rew_actor, d, info, kappa, beta))

        # at the brain again
        m, cum_rew_actor_delayed, d, info = self.receive_observation()
        r = cum_rew_actor_delayed - self.cum_rew_brain
        self.cum_rew_brain = cum_rew_actor_delayed

        self.t += 1

        # print("DEBUG: end of step ---")
        # print(f"DEBUG: self.past_actions:{self.past_actions}")
        # print(f"DEBUG: self.past_observations:{self.past_observations}")
        # print(f"DEBUG: self.arrival_times_actions:{self.arrival_times_actions}")
        # print(f"DEBUG: self.arrival_times_observations:{self.arrival_times_observations}")
        # print(f"DEBUG: self.t:{self.t}")
        # print("DEBUG: ---")
        return m, r, d, info

    def send_action(self, action, init=False):
        """
        Appends action to the left of self.past_actions
        Simulates the time at which it will reach the agent and stores it on the left of self.arrival_times_actions
        """
        # at the brain
        kappa, = sample(self.act_delay_range, 1) if not init else [0, ]  # TODO: change this if we implement a different initialization
        self.arrival_times_actions.appendleft(self.t + kappa)
        self.past_actions.appendleft(action)

    def receive_action(self):
        """
        Looks for the last created action that has arrived before t at the agent
        NB: since it is the most recently created action that the agent got, this is the one that is to be applied
        Returns:
            next_action_idx: int: the index of the action that is going to be applied
            prev_action_idx: int: the index of the action previously being applied (i.e. of the action that influenced the observation since it is retrieved instantaneously in usual Gym envs)
        """
        # CAUTION: from the brain point of view, the "previous action"'s age (kappa_t) is not like the previous "next action"'s age (beta_{t-1}) (e.g. repeated observations)
        prev_action_idx = self.prev_action_idx + 1  # + 1 is to account for the fact that this was the right idx 1 time-step before
        next_action_idx = next(i for i, t in enumerate(self.arrival_times_actions) if t <= self.t)
        self.prev_action_idx = next_action_idx
        self.next_action = self.past_actions[next_action_idx]
        # print(f"DEBUG: next_action_idx:{next_action_idx}, prev_action_idx:{prev_action_idx}")
        return next_action_idx, prev_action_idx

    def send_observation(self, obs):
        """
        Appends obs to the left of self.past_observations
        Simulates the time at which it will reach the brain and appends it in self.arrival_times_observations
        """
        # at the remote actor
        alpha, = sample(self.obs_delay_range, 1)
        self.arrival_times_observations.appendleft(self.t + alpha)
        self.past_observations.appendleft(obs)

    def receive_observation(self):
        """
        Looks for the last created observation at the agent/observer that reached the brain at time t
        NB: since this is the most recently created observation that the brain got, this is the one currently being considered as the last observation
        Returns:
            augmented_obs: tuple:
                m: object: last observation that reached the brain
                past_actions: tuple: the history of actions that the brain sent so far
                alpha: int: number of micro time steps it took the last observation to travel from the agent/observer to the brain
                kappa: int: action travel delay + number of micro time-steps for which the next action has been applied at the agent
                beta: int: action travel delay + number of micro time-steps for which the previous action has been applied at the agent
            r: float: delayed reward corresponding to the transition that created m
            d: bool: delayed done corresponding to the transition that created m
            info: dict: delayed info corresponding to the transition that created m
        """
        # at the brain
        alpha = next(i for i, t in enumerate(self.arrival_times_observations) if t <= self.t)
        m, r, d, info, kappa, beta = self.past_observations[alpha]
        return (m, tuple(itertools.islice(self.past_actions, 0, self.past_actions.maxlen - 1)), alpha, kappa, beta), r, d, info


class UnseenRandomDelayWrapper(RandomDelayWrapper):
    """
    Wrapper that translates the RandomDelayWrapper back to the usual RL setting
    Use this wrapper to see what happens to vanilla RL algorithms facing random delays
    """

    def __init__(self, env, **kwargs):
        super().__init__(env, **kwargs)
        self.observation_space = env.observation_space

    def reset(self, **kwargs):
        t = super().reset(**kwargs)  # t: (m, tuple(self.past_actions), alpha, kappa, beta)
        return t[0]

    def step(self, action):
        t, *aux = super().step(action)  # t: (m, tuple(self.past_actions), alpha, kappa, beta)
        return (t[0], *aux)


def simple_wifi_sampler1():
    return np.random.choice([1, 2, 3, 4, 5, 6], p=[0.3082, 0.5927, 0.0829, 0.0075, 0.0031, 0.0056])


def simple_wifi_sampler2():
    return np.random.choice([1, 2, 3, 4], p=[0.3082, 0.5927, 0.0829, 0.0162])


class WifiDelayWrapper1(RandomDelayWrapper):
    """
    Simple sampler built from a dataset of 10000 real-world wifi communications
    The atomic time-step is 0.02s
    All communication times above 0.1s have been clipped to 0.1s
    """

    def __init__(self, env, initial_action=None, skip_initial_actions=False):
        super().__init__(env, obs_delay_range=range(0, 7), act_delay_range=range(0, 7), initial_action=initial_action, skip_initial_actions=skip_initial_actions)

    def send_observation(self, obs):
        # at the remote actor
        alpha = simple_wifi_sampler1()
        self.arrival_times_observations.appendleft(self.t + alpha)
        self.past_observations.appendleft(obs)

    def send_action(self, action, init=False):
        # at the brain
        kappa = simple_wifi_sampler1() if not init else 0  # TODO: change this if we implement a different initialization
        self.arrival_times_actions.appendleft(self.t + kappa)
        self.past_actions.appendleft(action)


class WifiDelayWrapper2(RandomDelayWrapper):
    """
    Simple sampler built from a dataset of 10000 real-world wifi communications
    The atomic time-step is 0.02s
    All communication times above 0.1s have been clipped to 0.1s
    """

    def __init__(self, env, initial_action=None, skip_initial_actions=False):
        super().__init__(env, obs_delay_range=range(0, 5), act_delay_range=range(0, 5), initial_action=initial_action, skip_initial_actions=skip_initial_actions)

    def send_observation(self, obs):
        # at the remote actor
        alpha = simple_wifi_sampler2()
        self.arrival_times_observations.appendleft(self.t + alpha)
        self.past_observations.appendleft(obs)

    def send_action(self, action, init=False):
        # at the brain
        kappa = simple_wifi_sampler2() if not init else 0  # TODO: change this if we implement a different initialization
        self.arrival_times_actions.appendleft(self.t + kappa)
        self.past_actions.appendleft(action)
