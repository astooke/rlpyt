
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
from rlpyt.samplers.parallel.gpu.sampler import GpuParallelSampler
from rlpyt.samplers.parallel.gpu.collectors import (GpuResetCollector,
    GpuWaitResetCollector)
from rlpyt.envs.atari.atari_env import AtariEnv
from rlpyt.algos.pg.a2c import A2C
from rlpyt.agents.pg.atari import AtariLstmAgent
from rlpyt.runners.minibatch_rl import MinibatchRl
from rlpyt.utils.logging.context import logger_context


def build_and_train(game="pong", run_ID=0, cuda_idx=None, mid_batch_reset=False, n_parallel=2):
    affinity = dict(cuda_idx=cuda_idx, workers_cpus=list(range(n_parallel)))
    Collector = GpuResetCollector if mid_batch_reset else GpuWaitResetCollector
    print(f"To satisfy mid_batch_reset=={mid_batch_reset}, using {Collector}.")

    sampler = GpuParallelSampler(
        EnvCls=AtariEnv,
        env_kwargs=dict(game=game, num_img_obs=1),  # Learn on individual frames.
        CollectorCls=Collector,
        batch_T=20,  # Longer sampling/optimization horizon for recurrence.
        batch_B=16,  # 16 parallel environments.
        max_decorrelation_steps=400,
    )
    algo = A2C()  # Run with defaults.
    agent = AtariLstmAgent()
    runner = MinibatchRl(
        algo=algo,
        agent=agent,
        sampler=sampler,
        n_steps=50e6,
        log_interval_steps=1e5,
        affinity=affinity,
    )
    config = dict(game=game)
    name = "a2c_" + game
    log_dir = "example_4"
    with logger_context(log_dir, run_ID, name, config):
        runner.train()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--game', help='Atari game', default='pong')
    parser.add_argument('--run_ID', help='run identifier (logging)', type=int, default=0)
    parser.add_argument('--cuda_idx', help='gpu to use ', type=int, default=None)
    parser.add_argument('--mid_batch_reset', help='whether environment resets during itr',
        type=bool, default=False)
    parser.add_argument('--n_parallel', help='number of sampler workers', type=int, default=2)
    args = parser.parse_args()
    build_and_train(
        game=args.game,
        run_ID=args.run_ID,
        cuda_idx=args.cuda_idx,
        mid_batch_reset=args.mid_batch_reset,
        n_parallel=args.n_parallel,
    )
