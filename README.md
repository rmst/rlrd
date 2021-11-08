# Reinforcement Learning with Random Delays

PyTorch implementation of our paper [Reinforcement Learning with Random Delays (ICLR 2020)](https://openreview.net/forum?id=QFYnKlBJYR) – [[Arxiv]](https://arxiv.org/abs/2010.02966)

### Getting Started
This repository can be pip-installed via:
```bash
pip install git+https://github.com/rmst/rlrd.git
```

DC/AC can be run on a simple 1-step delayed `Pendulum-v0` task via:
```bash
python -m rlrd run rlrd:DcacTraining Env.id=Pendulum-v0
```

Hyperparameters can be set via command line. E.g.:
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


To train DC/AC on a 1-step delayed version of `HalfCheetah-v2`, run:
```bash
python -m rlrd run rlrd:DcacTraining Env.id=HalfCheetah-v2
```

To train SAC on a 1-step delayed version of `Ant-v2` run:
```bash
python -m rlrd run rlrd:DelayedSacTraining Env.id=Ant-v2
```

### Weights and Biases API
Your curves can be exported directly to the Weights and Biases (wandb) website by using `run-wandb`.
For example, to run DC/AC on Pendulum with a 1-step delay and export the curves to your wanb project:

```terminal
python -m rlrd run-wandb \
yourWandbID \
yourWandbProjectName \
aNameForTheWandbRun \
aFileNameForLocalCheckpoints \
rlrd:DcacTraining Env.id=Pendulum-v0
```

Use the optional hyperparameters descibed before to play with more meaningful delays.

### Contribute / known issues
Contributions are welcome.
Please submit a PR with your name in the contributors list.

We did not yet optimize our python implementation of DC/AC, this is the most important thing to do right now as it is quite slow.

In particular, a lot of time is wasted when artificially re-creating a batched tensor for computing the value estimates in one forward pass, and the replay buffer is inefficient.
See the `#FIXME` in [dcac.py](https://github.com/rmst/rlrd/blob/master/rlrd/dcac.py)
