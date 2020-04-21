import gym
import torch

from torch.nn import Linear, Sequential, ReLU, ModuleList, Module
from torch.nn import functional as F
from agents.sac_models import ActorModule
from agents.nn import TanhNormalLayer

from agents.envs import RandomDelayEnv


class DelayedBranchedMlpModule(Module):
    def __init__(self, observation_space, action_space, device, is_Q_network, hidden_units: int = 256):
        super().__init__()
        assert isinstance(observation_space, gym.spaces.Tuple)
        # TODO: check that it is actually an instance of:
        # Tuple((
        # 	obs_space,  # most recent observation
        # 	Tuple([act_space] * (obs_delay_range.stop + act_delay_range.stop)),  # action buffer
        # 	Discrete(obs_delay_range.stop),  # observation delay int64
        # 	Discrete(act_delay_range.stop),  # action delay int64
        # ))

        self.device = device
        self.is_Q_network = is_Q_network

        self.obs_dim = observation_space[0].shape[0]
        self.buf_size = len(observation_space[1])
        self.act_dim = observation_space[1][0].shape[0]

        self.lin_obs = Linear(self.obs_dim + self.buf_size, hidden_units)  # TODO: find a better solution
        self.lin_act = Linear(self.act_dim * self.buf_size + self.buf_size, hidden_units)  # TODO: find a better solution

        if self.is_Q_network:
            self.lin_merged = Linear(2 * hidden_units + action_space.shape[0], hidden_units)
        else:
            self.lin_merged = Linear(2 * hidden_units, hidden_units)

    def forward(self, x):
        assert isinstance(x, tuple)
        # TODO: check that x is actually in:
        # Tuple((
        # 	obs_space,  # most recent observation
        # 	Tuple([act_space] * (obs_delay_range.stop + act_delay_range.stop)),  # action buffer
        # 	Discrete(obs_delay_range.stop),  # observation delay int64
        # 	Discrete(act_delay_range.stop),  # action delay int64
        # ))

        # TODO: double check that everything is correct (dims, devices, autograd)

        # print(f"DEBUG: forward --------------------------")
        # print(f"DEBUG: is_Q_network:{self.is_Q_network}")
        # print(f"DEBUG:x[0].shape:{x[0].shape}")
        # print(f"DEBUG: len(x[1]):{len(x[1])}")
        # print(f"DEBUG:x[2]:{x[2]}")
        # print(f"DEBUG:x[3]:{x[3]}")
        obs = x[0]
        act_buf = torch.cat(x[1], dim=1)
        obs_del = x[2]
        act_del = x[3]
        if self.is_Q_network:
            act = x[4]  # TODO: check that this is correct

        batch_size = obs.shape[0]
        obs_one_hot = torch.zeros(batch_size, self.buf_size, device=self.device).scatter_(1, obs_del.unsqueeze(1), 1.0)  # TODO: check that scatter_ doesn't create a [1.0] tensor on CPU
        act_one_hot = torch.zeros(batch_size, self.buf_size, device=self.device).scatter_(1, act_del.unsqueeze(1), 1.0)  # TODO: check that scatter_ doesn't create a [1.0] tensor on CPU

        # print(f"DEBUG:obs:{obs}")
        # print(f"DEBUG:act_buf:{act_buf}")
        # print(f"DEBUG:obs_del:{obs_del}")
        # print(f"DEBUG:act_del:{act_del}")
        # print(f"DEBUG:obs_one_hot:{obs_one_hot}")
        # print(f"DEBUG:act_one_hot:{act_one_hot}")

        input_obs = torch.cat((obs, obs_one_hot), dim=1)
        input_act = torch.cat((act_buf, act_one_hot), dim=1)

        # print(f"DEBUG:input_obs.shape:{input_obs.shape}")
        # print(f"DEBUG:input_act.shape:{input_act.shape}")

        h_obs = F.relu(self.lin_obs(input_obs))
        h_act = F.relu(self.lin_act(input_act))

        # print(f"DEBUG:h_obs.shape:{h_obs.shape}")
        # print(f"DEBUG:h_act.shape:{h_act.shape}")

        if self.is_Q_network:
            h = torch.cat((h_obs, h_act, act), dim=1)
        else:
            h = torch.cat((h_obs, h_act), dim=1)

        # print(f"DEBUG:after merge h.shape:{h.shape}")

        h = self.lin_merged(h)

        # print(f"DEBUG:output subnet h.shape:{h.shape}")

        return h


class DelayedMlpModule(Module):
    def __init__(self, observation_space, action_space, device, is_Q_network, hidden_units: int = 256, obs_delay=True, act_delay=True):  # FIXME: action_space param is useless
        """
        Args:
            observation_space:
                Tuple((
                    obs_space,  # most recent observation
                    Tuple([act_space] * (obs_delay_range.stop + act_delay_range.stop)),  # action buffer
                    Discrete(obs_delay_range.stop),  # observation delay int64
                    Discrete(act_delay_range.stop),  # action delay int64
                ))
            action_space
            device: the device on which the variables are instanciated
            is_Q_network: bool: if True, the input of forward() expects the action to be appended at the end of the input
            hidden_units: number of output units of this module
            (optional) obs_delay: bool (default True): if False, the observation delay of observation_space will be ignored (e.g. unknown)
            (optional) act_delay: bool (default True): if False, the action delay of observation_space will be ignored (e.g. unknown)
        """
        super().__init__()
        assert isinstance(observation_space, gym.spaces.Tuple)
        # TODO: check that it is actually an instance of:
        # Tuple((
        # 	obs_space,  # most recent observation
        # 	Tuple([act_space] * (obs_delay_range.stop + act_delay_range.stop)),  # action buffer
        # 	Discrete(obs_delay_range.stop),  # observation delay int64
        # 	Discrete(act_delay_range.stop),  # action delay int64
        # ))

        self.device = device
        self.is_Q_network = is_Q_network
        self.act_delay = act_delay
        self.obs_delay = obs_delay

        self.obs_dim = observation_space[0].shape[0]
        self.buf_size = len(observation_space[1])
        self.act_dim = observation_space[1][0].shape[0]
        assert self.act_dim == action_space.shape[0], f"action spaces mismatch: {self.act_dim} and {action_space.shape[0]}"

        if self.is_Q_network:
            if self.act_delay and self.obs_delay:
                self.lin = Linear(self.obs_dim + (self.act_dim + 2) * self.buf_size + self.act_dim, hidden_units)
            elif self.act_delay or self.obs_delay:
                self.lin = Linear(self.obs_dim + (self.act_dim + 1) * self.buf_size + self.act_dim, hidden_units)
            else:
                self.lin = Linear(self.obs_dim + self.act_dim * self.buf_size + self.act_dim, hidden_units)
        else:
            if self.act_delay and self.obs_delay:
                self.lin = Linear(self.obs_dim + (self.act_dim + 2) * self.buf_size, hidden_units)
            elif self.act_delay or self.obs_delay:
                self.lin = Linear(self.obs_dim + (self.act_dim + 1) * self.buf_size, hidden_units)
            else:
                self.lin = Linear(self.obs_dim + self.act_dim * self.buf_size, hidden_units)

    def forward(self, x):
        assert isinstance(x, tuple)
        # TODO: check that x is actually in:
        # Tuple((
        # 	obs_space,  # most recent observation
        # 	Tuple([act_space] * (obs_delay_range.stop + act_delay_range.stop)),  # action buffer
        # 	Discrete(obs_delay_range.stop),  # observation delay int64
        # 	Discrete(act_delay_range.stop),  # action delay int64
        # ))

        # TODO: double check that everything is correct (dims, devices, autograd)

        # print(f"DEBUG: forward --------------------------")
        # print(f"DEBUG: is_Q_network:{self.is_Q_network}")
        # print(f"DEBUG:x[0].shape:{x[0].shape}")
        # print(f"DEBUG: len(x[1]):{len(x[1])}")
        # print(f"DEBUG:x[2]:{x[2]}")
        # print(f"DEBUG:x[3]:{x[3]}")

        obs = x[0]
        act_buf = torch.cat(x[1], dim=1)

        input = torch.cat((obs, act_buf), dim=1)

        batch_size = obs.shape[0]
        if self.obs_delay:
            obs_del = x[2]
            obs_one_hot = torch.zeros(batch_size, self.buf_size, device=self.device).scatter_(1, obs_del.unsqueeze(1), 1.0)  # TODO: check that scatter_ doesn't create a [1.0] tensor on CPU
            input = torch.cat((input, obs_one_hot), dim=1)
        if self.act_delay:
            act_del = x[3]
            act_one_hot = torch.zeros(batch_size, self.buf_size, device=self.device).scatter_(1, act_del.unsqueeze(1), 1.0)  # TODO: check that scatter_ doesn't create a [1.0] tensor on CPU
            input = torch.cat((input, act_one_hot), dim=1)
        if self.is_Q_network:
            act = x[4]  # TODO: check that this is correct
            input = torch.cat((input, act), dim=1)

        h = self.lin(input)

        return h


class MlpActionValue(Sequential):
    def __init__(self, observation_space, action_space, hidden_units, device, act_delay=True, obs_delay=True):
        super().__init__(
            DelayedMlpModule(observation_space, action_space, device, is_Q_network=True, act_delay=act_delay, obs_delay=obs_delay), ReLU(),
            Linear(hidden_units, hidden_units), ReLU(),
            Linear(hidden_units, 2)  # reward and entropy predicted separately
        )

    # noinspection PyMethodOverriding
    def forward(self, obs, action):
        x = (*obs, action)
        return super().forward(x)


class MlpPolicy(Sequential):
    def __init__(self, observation_space, action_space, hidden_units, device, act_delay=True, obs_delay=True):
        super().__init__(
            DelayedMlpModule(observation_space, action_space, device, is_Q_network=False, act_delay=act_delay, obs_delay=obs_delay), ReLU(),
            Linear(hidden_units, hidden_units), ReLU(),
            TanhNormalLayer(hidden_units, action_space.shape[0])
        )

    # noinspection PyMethodOverriding
    def forward(self, obs):
        return super().forward(obs)


class Mlp(ActorModule):
    def __init__(self, observation_space, action_space, hidden_units: int = 256, num_critics: int = 2, act_delay=False, obs_delay=False):
        super().__init__()
        assert isinstance(observation_space, gym.spaces.Tuple)
        self.critics = ModuleList(MlpActionValue(observation_space, action_space, hidden_units, self.device, act_delay=act_delay, obs_delay=obs_delay) for _ in range(num_critics))
        self.actor = MlpPolicy(observation_space, action_space, hidden_units, self.device, act_delay=act_delay, obs_delay=obs_delay)
        self.critic_output_layers = [c[-1] for c in self.critics]


# === Testing ==========================================================================================================

if __name__ == "__main__":
    from agents import Training, run
    from agents.util import partial
    from agents.sac import Agent

    Delayed_Sac_Test = partial(
        Training,
        epochs=3,
        rounds=5,
        steps=500,
        Agent=partial(Agent, device='cpu', memory_size=1000000, start_training=256, batchsize=4, Model=Mlp),
        Env=partial(RandomDelayEnv, min_observation_delay=0, sup_observation_delay=1, min_action_delay=0, sup_action_delay=1),  # RTRL setting, should get roughly the same behavior as SAC in RTRL
    )

    Sac_Test = partial(
        Training,
        epochs=3,
        rounds=5,
        steps=500,
        Agent=partial(Agent, device='cpu', memory_size=1000000, start_training=256, batchsize=4),
        Env=partial(id="Pendulum-v0", real_time=True),
    )

    print("--- NOW RUNNING: SAC, delayed wrapper, delayed MLP model, RTRL setting ---")
    run(Delayed_Sac_Test)
    print("--- NOW RUNNING: SAC, normal env, normal MLP model, RTRL setting ---")
    run(Sac_Test)
