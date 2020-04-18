
import numpy as np
import gym
import pickle
import multiprocessing as mp
import time
import torch


USE_MP = False

def _step(args):
    env, action = args
    return env.step(action)


class BatchEnv:
    def __init__(self, internal_bs=64, num_procs=None):
        self.internal_bs = internal_bs
        self.envs = None
        self.pool = mp.Pool()

    def init_from_pickle(self, envs):
        self.envs = []
        for e in envs:
            self.envs += [e] * self.internal_bs
        if USE_MP:
            self.envs = self.pool.map(pickle.loads, self.envs)
        else:
            self.envs = list(map(pickle.loads, self.envs))

    def step(self, actions):
        if len(actions) < len(self.envs):
            # first action
            if isinstance(actions, torch.Tensor):
                actions = torch.repeat_interleave(actions, self.internal_bs, 0)
            elif isinstance(actions, np.ndarray):
                actions = actions.repeat(self.internal_bs, 0)
            else:
                raise TypeError('actions should be torch.tensor or numpy.ndarray')
        if USE_MP:
            res = self.pool.map(_step, zip(self.envs, actions))
        else:
            res = list(map(_step, zip(self.envs, actions)))

        obss, rewards, dones, info = [], [], [], []
        for r in res:
            obss.append(r[0])
            rewards.append(r[1])
            dones.append(r[2])
            info.append(r[3])
        return obss, rewards, dones, info



if __name__ == '__main__':

    bs = 32
    num_avg = 16
    num_steps = 8

    e = gym.make('Ant-v2')
    e.reset()

    e_pck = [pickle.dumps(e)] * bs # serialized envs

    # dummy policy
    def policy(num_agents):
        return np.array([e.action_space.sample() for _ in range(num_agents)])

    be = BatchEnv(num_avg)


    t = time.time()

    be.init_from_pickle(e_pck)

    actions = policy(bs) # first action
    obss, rewards, dones, _ = be.step(actions)

    for _ in range(num_steps):
        actions = policy(len(obss))
        be.step(actions)


    print(time.time() - t)

