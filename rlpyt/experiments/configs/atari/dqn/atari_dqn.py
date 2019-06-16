
import copy

from rlpyt.samplers.cpu.collectors import NoWaitResetCollector as CpuNwrCollector
from rlpyt.samplers.gpu.collectors import NoWaitResetCollector as GpuNwrCollector
from rlpyt.replays.non_sequence.full_n_step_frame_uniform import MonolithicUniformReplayFrameBuffer
from rlpyt.replays.non_sequence.uniform import UniformReplayBuffer
from rlpyt.replays.non_sequence.prioritized import PrioritizedReplayBuffer

configs = dict()


config = dict(
    agent=dict(),
    algo=dict(
        discount=0.99,
        batch_size=128,
        learning_rate=1e-4,  # Trying lower for adam.
        clip_grad_norm=10.,
        min_steps_learn=int(5e4),
        double_dqn=False,
        prioritized_replay=False,
        n_step_return=1,
        ReplayBufferCls=None,  # None selects frame buffer by replay option.
    ),
    env=dict(
        game="pong",
        episodic_lives=True,
    ),
    eval_env=dict(
        game="pong",  # NOTE: update in train script!
        episodic_lives=False,
        horizon=int(27e3),
    ),
    model=dict(dueling=False),
    optim=dict(),
    runner=dict(
        n_steps=50e6,
        log_interval_steps=1e6,
    ),
    sampler=dict(
        batch_T=2,
        batch_B=16,
        max_decorrelation_steps=1000,
        eval_n_envs=4,
        eval_max_steps=int(125e3),
        eval_max_trajectories=100,
        eval_min_envs_reset=1,
        CollectorCls=None,
    ),
)

configs["dqn"] = config

config = copy.deepcopy(config)
config["algo"]["double_dqn"] = True
config["algo"]["prioritized_replay"] = True
config["model"]["dueling"] = True

configs["double_pri_duel"] = config

config = copy.deepcopy(configs["dqn"])
config["algo"]["learning_rate"] = 2.5e-4
configs["catdqn"] = config

config = copy.deepcopy(config)
config["algo"]["n_step_return"] = 3
config["algo"]["learning_rate"] = 6.25e-5
configs["ernbw"] = config

config = copy.deepycopy(configs["dqn"])
config["algo"]["CollectorCls"] = GpuNwrCollector
configs["gpu_nowaitreset"] = config

config = copy.deepcopy(configs["dqn"])
configs["algo"]["CollectorCls"] = CpuNwrCollector
configs["cpu_nowaitreset"] = config

config = copy.deepcopy(configs["dqn"])
configs["algo"]["ReplayBufferCls"] = MonolithicUniformReplayFrameBuffer
configs["monolithic_uniformframe"] = config

config = copy.deepcopy(configs["dqn"])
config["algo"]["ReplayBufferCls"] = UniformReplayBuffer
configs["uniform_noframe"] = config

