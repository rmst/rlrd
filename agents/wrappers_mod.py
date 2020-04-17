from collections import Sequence, Mapping, deque
from random import randint, randrange
import itertools

import gym
import numpy as np
from gym.spaces import Tuple, Discrete
from gym.wrappers import TimeLimit


class RealTimeWrapper(gym.Wrapper):
	def __init__(self, env):
		super().__init__(env)
		self.observation_space = gym.spaces.Tuple((env.observation_space, env.action_space))
		# self.initial_action = env.action_space.sample()
		assert isinstance(env.action_space, gym.spaces.Box)
		self.initial_action = env.action_space.high * 0

	def reset(self):
		self.previous_action = self.initial_action
		return super().reset(), self.previous_action

	def step(self, action):
		observation, reward, done, info = super().step(self.previous_action)
		self.previous_action = action
		return (observation, action), reward, done, info


class PreviousActionWrapper(gym.Wrapper):
	def __init__(self, env):
		super().__init__(env)
		self.observation_space = gym.spaces.Tuple((env.observation_space, env.action_space))
		# self.initial_action = env.action_space.sample()
		assert isinstance(env.action_space, gym.spaces.Box)
		self.initial_action = env.action_space.high * 0

	def reset(self):
		self.previous_action = self.initial_action
		return super().reset(), self.previous_action

	def step(self, action):
		observation, reward, done, info = super().step(action)  # this line is different from RealTimeWrapper
		self.previous_action = action
		return (observation, action), reward, done, info


class StatsWrapper(gym.Wrapper):
	"""Compute running statistics (return, number of episodes, etc.) over a certain time window."""

	def __init__(self, env, window=100):
		super().__init__(env)
		self.reward_hist = deque([0], maxlen=window + 1)
		self.done_hist = deque([1], maxlen=window + 1)
		self.total_steps = 0

	def reset(self, **kwargs):
		return super().reset(**kwargs)

	def step(self, action):
		m, r, d, info = super().step(action)
		self.reward_hist.append(r)
		self.done_hist.append(d)
		self.total_steps += 1
		return m, r, d, info

	def stats(self):
		returns = [0]
		steps = [0]
		for reward, done in zip(self.reward_hist, self.done_hist):
			returns[-1] += reward
			steps[-1] += 1
			if done:
				returns.append(0)
				steps.append(0)
		returns = returns[1:-1]  # first and last episodes are incomplete
		steps = steps[1:-1]

		return dict(
			episodes=len(returns),
			episode_length=np.mean(steps) if len(steps) else np.nan,
			returns=np.mean(returns) if len(returns) else np.nan,
			average_reward=np.mean(tuple(self.reward_hist)[1:]),
		)


class DictObservationWrapper(gym.ObservationWrapper):
	def __init__(self, env, key='vector'):
		super().__init__(env)
		self.key = key
		self.observation_space = gym.spaces.Dict({self.key: env.observation_space})

	def observation(self, observation):
		return {self.key: observation}


class TupleObservationWrapper(gym.ObservationWrapper):
	def __init__(self, env):
		super().__init__(env)
		self.observation_space = gym.spaces.Tuple((env.observation_space,))

	def observation(self, observation):
		return observation,


class DictActionWrapper(gym.Wrapper):
	def __init__(self, env, key='value'):
		super().__init__(env)
		self.key = key
		self.action_space = gym.spaces.Dict({self.key: env.action_space})

	def step(self, action: dict):
		return self.env.step(action['value'])


class AffineObservationWrapper(gym.ObservationWrapper):
	def __init__(self, env, shift, scale):
		super().__init__(env)
		assert isinstance(env.observation_space, gym.spaces.Box)
		self.shift = shift
		self.scale = scale
		self.observation_space = gym.spaces.Box(self.observation(env.observation_space.low), self.observation(env.observation_space.high), dtype=env.observation_space.dtype)

	def observation(self, obs):
		return (obs + self.shift) * self.scale


class AffineRewardWrapper(gym.RewardWrapper):
	def __init__(self, env, shift, scale):
		super().__init__(env)
		self.shift = shift
		self.scale = scale

	def reward(self, reward):
		return (reward + self.shift) / self.scale


class NormalizeActionWrapper(gym.Wrapper):
	def __init__(self, env):
		super().__init__(env)
		self.scale = env.action_space.high - env.action_space.low
		self.shift = env.action_space.low
		self.action_space = gym.spaces.Box(-np.ones_like(self.shift), np.ones_like(self.shift), dtype=env.action_space.dtype)

	def reset(self, **kwargs):
		return self.env.reset(**kwargs)

	def step(self, action):
		action = action / 2 + 0.5  # 0 < a < 1
		action = action * self.scale + self.shift
		return self.env.step(action)


class TimeLimitResetWrapper(gym.Wrapper):
	"""Adds a `reset` key to `info` that indicates whether an episode was ended just because of a time limit.
	This can be important as running out of time, should usually not be considered a "true" terminal state."""

	def __init__(self, env, max_steps=None, key='reset'):
		super().__init__(env)
		self.reset_key = key
		self.enforce = bool(max_steps)
		if max_steps is None:
			tl = get_wrapper_by_class(env, TimeLimit)
			max_steps = 1 << 31 if tl is None else tl._max_episode_steps
			# print("TimeLimitResetWrapper.max_steps =", max_steps)

		self.max_steps = max_steps
		self.t = 0

	def reset(self, **kwargs):
		m = self.env.reset(**kwargs)
		self.t = 0
		return m

	def step(self, action):
		m, r, d, info = self.env.step(action)

		reset = (self.t == self.max_steps - 1) or info.get(self.reset_key, False)
		if not self.enforce:
			if reset:
				assert d, f"something went wrong t={self.t}, max_steps={self.max_steps}, info={info}"
		else:
			d = d or reset
		info = {**info, self.reset_key: reset}
		self.t += 1
		return m, r, d, info


class Float64ToFloat32(gym.ObservationWrapper):
	"""Converts np.float64 arrays in the observations to np.float32 arrays."""

	# TODO: change observation/action spaces to correct dtype
	def observation(self, observation):
		observation = deepmap({np.ndarray: float64_to_float32}, observation)
		return observation

	def step(self, action):
		s, r, d, info = super().step(action)
		return s, r, d, info


class FrameSkip(gym.Wrapper):
	def __init__(self, env, n, rs=1):
		assert n >= 1
		super().__init__(env)
		self.frame_skip = n
		self.reward_scale = rs

	def step(self, action):
		reward = 0
		for i in range(self.frame_skip):
			m, r, d, info = self.env.step(action)
			reward += r
			if d:
				break
		return m, reward * self.reward_scale, d, info


class RandomDelayWrapper(gym.Wrapper):
	"""
	Wrapper for any environment modelling random observation and action delays
	Note that you can access most recent action known to be applied with past_actions[action_delay + observation_delay]
	NB: action_delay represents the inference time + action channel delay + number of time-steps for which it has been applied
		The brain only needs this information to identify the action that was being executed when the observation was captured
		(the duration for which it had already been executed is irrelevant in the Markov assumption)
	Kwargs:
		max_observation_delay: int (default 8): maximum number of micro time steps it takes an observation to travel in the observation channel
		min_observation_delay: int (default 0): minimum number of micro time steps it takes an observation to travel in the observation channel
		max_action_delay: int (default 2): maximum number of micro time steps it takes an action to be computed and travel in the action channel
		min_action_delay: int (default 0): minimum number of micro time steps it takes an action to be computed and travel in the action channel
		instant_rewards: bool (default True): whether to send instantaneous step rewards (True) or delayed rewards (False)
		initial_action: action (default None): action with which the action buffer is filled at reset() (if None, sampled in the action space)
	"""

	def __init__(self, env, max_observation_delay=8, min_observation_delay=0, max_action_delay=2, min_action_delay=0, instant_rewards: bool = True, initial_action=None):
		super().__init__(env)
		assert 0 <= min_observation_delay <= max_observation_delay
		assert 0 <= min_action_delay <= max_action_delay
		self.instant_rewards = instant_rewards
		self.max_action_delay = max_action_delay
		self.max_observation_delay = max_observation_delay
		self.min_action_delay = min_action_delay
		self.min_observation_delay = min_observation_delay

		self.observation_space = Tuple((
			env.observation_space,  # most recent observation
			Tuple([env.action_space] * (max_observation_delay + max_action_delay)),  # action buffer
			Discrete(max_observation_delay),  # observation delay int64
			Discrete(max_action_delay),  # action delay int64
		))

		self.initial_action = initial_action
		# +1 everywhere to handle 0 delays, the additional slot is removed when sending observation
		self.past_actions = deque(maxlen=max_observation_delay + max_action_delay + 1)
		self.past_observations = deque(maxlen=max_observation_delay + 1)
		self.arrival_times_actions = deque(maxlen=max_action_delay + 1)
		self.arrival_times_observations = deque(maxlen=max_observation_delay + 1)

		self.t = 0
		self.done_signal_sent = False
		self.current_action = None

	def reset(self, **kwargs):
		self.done_signal_sent = False
		first_observation = super().reset(**kwargs)

		# fill up buffers
		self.t = - (self.max_observation_delay + self.max_action_delay + 1)
		while self.t < 0:
			act = self.action_space.sample() if self.initial_action is None else self.initial_action
			self.send_action(act)
			self.send_observation((first_observation, 0., False, {}, 0))
			self.t += 1

		assert self.t == 0
		received_observation, *_ = self.receive_observation()
		print("DEBUG: end of reset ---")
		print(f"DEBUG: self.past_actions:{self.past_actions}")
		print(f"DEBUG: self.past_observations:{self.past_observations}")
		print(f"DEBUG: self.arrival_times_actions:{self.arrival_times_actions}")
		print(f"DEBUG: self.arrival_times_observations:{self.arrival_times_observations}")
		print(f"DEBUG: self.t:{self.t}")
		print("DEBUG: ---")
		return received_observation

	def step(self, action):
		"""
		When both action and observation delays are 0, this step function is equivalent to the usual RL setting:
			The brain computes a new action instantly
			It sends it to the agent instantly
			The environment steps with this action
			The agents sends a new observation instantly to the brain
		When action delay is 1 and observation delay is 0, this is equivalent to the RTRL setting
		(The inference time is considered part of the action_delay)
		"""

		# at the brain
		self.send_action(action)

		# at the remote actor
		# if not self.first_act_received:
		if self.t < self.max_action_delay:
			# do nothing until the brain's first actions arrive at the remote actor
			self.receive_action()
			aux = 0, False, {}
		elif self.done_signal_sent:
			# just resend the last observation until the brain gets it
			self.send_observation(self.past_observations[0])
			aux = 0, False, {}
		else:
			cur_action_age = self.receive_action()
			print(f"DEBUG: curent_action sent to step:{self.current_action}")
			m, *aux = self.env.step(self.current_action)  # TODO: check that this is correct (it was before receive_action())
			self.send_observation((m, *aux, cur_action_age))

		# at the brain again
		m, *delayed_aux = self.receive_observation()
		aux = aux if self.instant_rewards else delayed_aux
		print("DEBUG: end of step ---")
		print(f"DEBUG: self.past_actions:{self.past_actions}")
		print(f"DEBUG: self.past_observations:{self.past_observations}")
		print(f"DEBUG: self.arrival_times_actions:{self.arrival_times_actions}")
		print(f"DEBUG: self.arrival_times_observations:{self.arrival_times_observations}")
		print(f"DEBUG: self.t:{self.t}")
		print("DEBUG: ---")
		self.t += 1
		return (m, *aux)

	def send_action(self, action):
		"""
		Appends action on the left of self.past_actions
		Simulates the time at which it will reach the agent and stores it on the left of self.arrival_times_actions
		"""
		# at the brain
		self.arrival_times_actions.appendleft(self.t + randrange(self.min_action_delay, self.max_action_delay + 1))  # TODO: could be any distribution  # EDIT: randrange goes from a to b - 1
		self.past_actions.appendleft(action)

	def receive_action(self):
		"""
		Looks for the last created action that has arrived before t at the agent
		NB: since this is the most recently created action that the agent got, this is the one currently being applied
		Returns:
			applied_action: int: the index of the action currently being applied
		"""
		applied_action = next(i for i, t in enumerate(self.arrival_times_actions) if t <= self.t)
		self.current_action = self.past_actions[applied_action]
		# if not self.first_act_received:
		# 	print(f"DEBUG: applied_action:{applied_action}, self.t:{self.t}")
		# 	self.first_act_received = (applied_action >= self.t)
		return applied_action

	def send_observation(self, obs):
		"""
		Appends obs on the left of self.past_observations
		Simulates the time at which it will reach the brain and appends it in self.arrival_times_observations
		"""
		# at the remote actor
		self.past_observations.appendleft(obs)
		self.arrival_times_observations.appendleft(self.t + randrange(self.min_observation_delay, self.max_observation_delay + 1))  # TODO: could be any distribution  # EDIT: randrange goes from a to b - 1

	def receive_observation(self):
		"""
		Looks for the last created observation at the agent/observer that reached the brain at time t
		NB: since this is the most recently created observation that the brain got, this is the one currently being considered as the last observation
		Returns:
			augmented_obs: tuple:
				m: object: last observation that reached the brain
				past_actions: tuple: the history of actions that the brain sent so far
				observation_delay: int: number of micro time steps it took the last observation to travel from the agent/observer to the brain
				action_delay: int: action travel delay + number of micro time-steps for which the action has been applied at the agent
			r: float: delayed reward corresponding to the transition that created m
			d: bool: delayed done corresponding to the transition that created m
			info: dict: delayed info corresponding to the transition that created m
		"""
		# at the brain
		observation_delay = next(i for i, t in enumerate(self.arrival_times_observations) if t <= self.t)
		m, r, d, info, action_delay = self.past_observations[observation_delay]
		return (m, tuple(itertools.islice(self.past_actions, 0, self.past_actions.maxlen - 1)), observation_delay, action_delay), r, d, info


class UnseenRandomDelayWrapper(RandomDelayWrapper):
	"""
	Wrapper that translates the RandomDelayWrapper back to the usual RL setting
	Use this wrapper to see what happens to vanilla RL algorithms facing random delays
	"""

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

	def reset(self, **kwargs):
		t = super().reset(**kwargs)  # t: (m, tuple(self.past_actions), observation_delay, action_delay)
		return t[0]

	def step(self, action):
		t, *aux = super().step(action)  # t: (m, tuple(self.past_actions), observation_delay, action_delay)
		return (t[0], *aux)


# === Utilities ========================================================================================================

def get_wrapper_by_class(env, cls):
	if isinstance(env, cls):
		return env
	elif isinstance(env, gym.Wrapper):
		return get_wrapper_by_class(env.env, cls)


def deepmap(f, m):
	"""Apply functions to the leaves of a dictionary or list, depending type of the leaf value.
	Example: deepmap({torch.Tensor: lambda t: t.detach()}, x)."""
	for cls in f:
		if isinstance(m, cls):
			return f[cls](m)
	if isinstance(m, Sequence):
		return type(m)(deepmap(f, x) for x in m)
	elif isinstance(m, Mapping):
		return type(m)((k, deepmap(f, m[k])) for k in m)
	else:
		raise AttributeError()


def float64_to_float32(x):
	return np.asarray(x, np.float32) if x.dtype == np.float64 else x
