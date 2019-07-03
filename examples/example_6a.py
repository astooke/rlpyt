
"""
Runs multiple instances of the Atari environment and optimizes using A2C
algorithm and a recurrent agent. Uses GPU parallel sampler, with option for
whether to reset environments in middle of sampling batch.

Standard recurrent agents cannot train with a reset in the middle of a
sequence, so all data after the environment 'done' signal will be ignored (see
variable 'valid' in algo).  So it may be preferable to pause those environments
and wait to reset them for the beginning of the next iteration.

If the environment takes a long time to reset relative to step, this may also
give a slight speed boost, as resets will happen in the workers while the master
is optimizing.  Feedforward agents are compatible with this arrangement by same
use of 'valid' mask.

"""
import sys

from rlpyt.utils.launching.affinity import get_affinity
from rlpyt.samplers.gpu.parallel_sampler import GpuParallelSampler
from rlpyt.samplers.gpu.collectors import WaitResetCollector
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

    affinity = get_affinity(slot_affinity_code)
    variant = load_variant(log_dir)
    global config
    config = update_config(config, variant)

    sampler = GpuParallelSampler(
        EnvCls=AtariEnv,
        env_kwargs=config["env"],
        CollectorCls=WaitResetCollector,
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
