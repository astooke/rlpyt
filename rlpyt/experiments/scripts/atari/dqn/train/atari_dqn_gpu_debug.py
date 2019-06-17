
import sys

from rlpyt.utils.launching.affinity import get_affinity
from rlpyt.samplers.gpu.parallel_sampler import GpuParallelSampler
from rlpyt.samplers.gpu.collectors import WaitResetCollector, ResetCollector
from rlpyt.envs.atari.atari_env import AtariEnv, AtariTrajInfo
from rlpyt.algos.dqn.dqn import DQN
from rlpyt.agents.dqn.atari.atari_dqn_agent import AtariDqnAgent
from rlpyt.runners.minibatch_rl_eval import MinibatchRlEval
from rlpyt.utils.logging.context import logger_context
from rlpyt.utils.launching.variant import load_variant, update_config

from rlpyt.replays.non_sequence.full_n_step_frame_uniform import MonolithicUniformReplayFrameBuffer
from rlpyt.replays.non_sequence.uniform import UniformReplayBuffer
from rlpyt.replays.non_sequence.prioritized import PrioritizedReplayBuffer

from rlpyt.experiments.configs.atari.dqn.atari_dqn_debug import configs


def build_and_train(slot_affinity_code, log_dir, run_ID, config_key):
    affinity = get_affinity(slot_affinity_code)
    config = configs[config_key]
    variant = load_variant(log_dir)
    config = update_config(config, variant)
    config["eval_env"]["game"] = config["env"]["game"]

    collector = config["sampler"].pop("collector", None)
    if collector == "reset_collector":
        CollectorCls = ResetCollector
        print("Using Reset Collector!")
    else:
        CollectorCls = WaitResetCollector
    sampler = GpuParallelSampler(
        EnvCls=AtariEnv,
        env_kwargs=config["env"],
        CollectorCls=CollectorCls,
        TrajInfoCls=AtariTrajInfo,
        eval_env_kwargs=config["eval_env"],
        **config["sampler"]
    )
    config["sampler"]["collector"] = collector
    replay_buffer = config["algo"].pop("replay_buffer", None)
    if replay_buffer == "monolithic_uniform_frame":
        ReplayBufferCls = MonolithicUniformReplayFrameBuffer
        print("Using monolithic_uniform_frame!")
    elif replay_buffer == "uniform_noframe":
        ReplayBufferCls = UniformReplayBuffer
        print("Using uniiform_noframe!")
    elif replay_buffer == "prioritized_noframe":
        ReplayBufferCls = PrioritizedReplayBuffer
    else:
        ReplayBufferCls = None

    algo = DQN(optim_kwargs=config["optim"], ReplayBufferCls=ReplayBufferCls,
        **config["algo"])
    config["algo"]["replay_buffer"] = replay_buffer
    agent = AtariDqnAgent(model_kwargs=config["model"], **config["agent"])
    runner = MinibatchRlEval(
        algo=algo,
        agent=agent,
        sampler=sampler,
        affinity=affinity,
        **config["runner"]
    )
    name = config["env"]["game"]
    with logger_context(log_dir, run_ID, name, config):
        runner.train()


if __name__ == "__main__":
    build_and_train(*sys.argv[1:])
