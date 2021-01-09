# RLRD

[Reinforcement Learning with Random Delays](https://arxiv.org/abs/2010.02966) in Pytorch

### Getting Started
This repository can be pip-installed via:
```bash
pip install git+https://github.com/rmst/rlrd.git
```

DCAC can be run on a simple 1-step delayed `Pendulum-v0` task via:
```bash
python -m rlrd run rlrd:DcacTraining Env.id=Pendulum-v0
```

Many optional hyperparameters can be set via command line. For instance:
```bash
python -m rlrd run rlrd:DcacTraining \
Env.id=Pendulum-v0 \
Env.min_observation_delay=0 \
Env.sup_observation_delay=2 \
Env.min_action_delay=0 \
Env.sup_action_delay=3 \
Agent.batchsize=128 \
Agent.memory_size=1000000 \
Agent.lr=0.0003 \
Agent.discount=0.99 \
Agent.target_update=0.005 \
Agent.reward_scale=5.0 \
Agent.entropy_scale=1.0 \
Agent.start_training=10000 \
Agent.device=cuda \
Agent.training_steps=1.0 \
Agent.loss_alpha=0.2 \
Agent.Model.hidden_units=256 \
Agent.Model.num_critics=2
```

Note that our gym wrapper adds a constant 1-step delay to the action delay, i.e. ```Env.min_action_delay=0``` actually means that the minimum action delay is 1 whereas ```Env.min_observation_delay=0``` means that the minimum observation delay is 0 (we assume that the action delay cannot be less than 1 time-step, e.g. for action inference).
For instance:
- ```Env.min_observation_delay=0 Env.sup_observation_delay=2``` means that the observation delay is randomly 0 or 1.
- ```Env.min_action_delay=0 Env.sup_action_delay=2``` means that the action delay is randomly 1 or 2.
- ```Env.min_observation_delay=1 Env.sup_observation_delay=2``` means that the observation delay is always 1.
- ```Env.min_observation_delay=0 Env.sup_observation_delay=3``` means that the observation delay is randomly 0, 1 or 2.
- etc.


### Mujoco Experiments
To install Mujoco, follow the instructions at [openai/gym](https://github.com/openai/gym).
The following environments were used in the paper:

![MuJoCo](resources/mujoco_horizontal.png)


To train DCAC on `HalfCheetah-v2` run:
```bash
python -m rlrd run rlrd:DcacTraining Env.id=HalfCheetah-v2
```

To SAC agent on `Ant-v2` run:
```bash
python -m rlrd run rlrd:DelayedSacTraining Env.id=HalfCheetah-v2
```

### Authors
- Yann Bouteiller
- Simon Ramstedt
