
import sys

from rlpyt.utils.launching.affinity import affinity_from_code
from rlpyt.samplers.cpu.parallel_sampler import CpuParallelSampler
from rlpyt.samplers.cpu.episodic_lives_collectors import EpisodicLivesWaitResetCollector
from rlpyt.envs.atari.atari_env import AtariEnv, AtariTrajInfo
from rlpyt.algos.policy_gradient.a2c import A2C
from rlpyt.agents.policy_gradient.atari.atari_ff_agent import AtariFfAgent
from rlpyt.runners.minibatch_rl import MinibatchRl
from rlpyt.utils.logging.context import logger_context
from rlpyt.utils.launching.variant import load_variant, update_config

from rlpyt.experiments.configs.atari.pg.atari_ff_a2c import configs


def build_and_train(slot_affinity_code, log_dir, run_ID, config_key):
    affinity = affinity_from_code(slot_affinity_code)
    config = configs[config_key]
    variant = load_variant(log_dir)
    config = update_config(config, variant)

    sampler = CpuParallelSampler(
        EnvCls=AtariEnv,
        env_kwargs=config["env"],
        CollectorCls=EpisodicLivesWaitResetCollector,
        TrajInfoCls=AtariTrajInfo,
        **config["sampler"]
    )
    algo = A2C(optim_kwargs=config["optim"], **config["algo"])
    agent = AtariFfAgent(model_kwargs=config["model"], **config["agent"])
    runner = MinibatchRl(
        algo=algo,
        agent=agent,
        sampler=sampler,
        affinity=affinity,
        **config["runner"]
    )
    name = config["env"]["game"]
    with logger_context(log_dir, run_ID, name, config):  # Might have to flatten config
        runner.train()


if __name__ == "__main__":
    build_and_train(*sys.argv[1:])
