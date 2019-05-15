
import sys
import json

from rlpyt.util.launching.affinity import get_affinity
from rlpyt.samplers.cpu.parallel_sampler import CpuParallelSampler
from rlpyt.samplers.cpu.collectors import EpisodicLivesWaitResetCollector
from rlpyt.envs.atari.atari_env import AtariEnv
from rlpyt.algos.policy_gradient.a2c import A2C
from rlpyt.agents.policy_gradient.atari.atari_ff_agent import AtariFfAgent
from rlpyt.runners.minibatch_rl import MinibatchRl
from rlpyt.utils.logging.context import logger_context

from rlpyt.scripts.atari.pg.config.atari_ff_a2c_cpu import default_configs


def build_and_run(run_slot_affinities_code, default_config_key,
        override_config_file, log_dir, game, run_ID, override_config_idx):
    affinities = get_affinity(run_slot_affinities_code)

    config = load_config(default_configs[default_config_key],
        override_config_file, override_config_idx)

    config["env"]["game"] = game
    sampler = CpuParallelSampler(
        EnvCls=AtariEnv,
        env_kwargs=config["env"],
        CollectorCls=EpisodicLivesWaitResetCollector,
        **config["sampler"]
    )

    algo = A2C(optim_kwargs=config["optim"], **config["algo"])
    agent = AtariFfAgent(model_kwargs=config["model"], **config["agent"])

    runner = MinibatchRl(
        algo=algo,
        agent=agent,
        sampler=sampler,
        affinities=affinities,
        **config["runner"]
    )

    with logger_context(log_dir, game, run_ID, config):
        runner.train()


if __name__ == "__main__":
    build_and_run(*sys.argv[1:])


def load_config(config, override_config_file, override_config_idx):
    override_config = json.load(override_config_file)[override_config_idx]
    for k, v in config:
        if k in override_config:
            v.update(override_config[k])
    return config
