
"""
Runs multiple instances of the Atari environment and optimizes using A2C
algorithm and a feed-forward agent. Uses GPU parallel sampler, with option for
whether to reset environments in middle of sampling batch.


"""
import sys

from rlpyt.utils.launching.affinity import affinity_from_code
from rlpyt.samplers.parallel.gpu.sampler import GpuSampler
from rlpyt.samplers.parallel.gpu.collectors import GpuWaitResetCollector
from rlpyt.envs.atari.atari_env import AtariEnv
from rlpyt.algos.pg.a2c import A2C
from rlpyt.agents.pg.atari import AtariFfAgent
from rlpyt.runners.minibatch_rl import MinibatchRl
from rlpyt.utils.logging.context import logger_context
from rlpyt.utils.launching.variant import load_variant, update_config


def build_and_train(slot_affinity_code, log_dir, run_ID):
    # (Or load from a central store of configs.)
    config = dict(
        env=dict(game="pong"),
        algo=dict(learning_rate=7e-4),
        sampler=dict(batch_B=16),
    )

    affinity = affinity_from_code(slot_affinity_code)
    variant = load_variant(log_dir)
    global config
    config = update_config(config, variant)

    sampler = GpuSampler(
        EnvCls=AtariEnv,
        env_kwargs=config["env"],
        CollectorCls=GpuWaitResetCollector,
        batch_T=5,
        # batch_B=16,  # Get from config.
        max_decorrelation_steps=400,
        **config["sampler"]
    )
    algo = A2C(**config["algo"])  # Run with defaults.
    agent = AtariFfAgent()
    runner = MinibatchRl(
        algo=algo,
        agent=agent,
        sampler=sampler,
        n_steps=50e6,
        log_interval_steps=1e5,
        affinity=affinity,
    )
    name = "a2c_" + config["env"]["game"]
    log_dir = "example_6"
    with logger_context(log_dir, run_ID, name, config):
        runner.train()


if __name__ == "__main__":
    build_and_train(*sys.argv[1:])
