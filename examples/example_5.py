
"""
Parallel sampler version of Atari DQN.  Increasing the number of parallel
environmnets (sampler batch_B) should improve the efficiency of the forward
pass for action sampling on the GPU.  Using a larger batch size in the algorithm
should improve the efficiency of the forward/backward passes during training.
(But both settings may impact hyperparameter selection and learning.)

"""

from rlpyt.samplers.parallel.gpu.sampler import GpuParallelSampler
from rlpyt.samplers.parallel.gpu.collectors import GpuWaitResetCollector
from rlpyt.envs.atari.atari_env import AtariEnv
from rlpyt.algos.dqn.dqn import DQN
from rlpyt.agents.dqn.atari.atari_dqn_agent import AtariDqnAgent
from rlpyt.runners.minibatch_rl import MinibatchRlEval
from rlpyt.utils.logging.context import logger_context


def build_and_train(game="pong", run_ID=0, cuda_idx=None, n_parallel=2):
    config = dict(
        env=dict(game=game),
        algo=dict(batch_size=128),
        sampler=dict(batch_T=2, batch_B=32),
    )
    sampler = GpuParallelSampler(
        EnvCls=AtariEnv,
        env_kwargs=dict(game=game),
        CollectorCls=GpuWaitResetCollector,
        eval_env_kwargs=dict(game=game),
        max_decorrelation_steps=0,
        eval_n_envs=10,
        eval_max_steps=int(10e3),
        eval_max_trajectories=5,
        # batch_T=4,  # Get from config.
        # batch_B=1,
        **config["sampler"]  # More parallel environments for batched forward-pass.
    )
    algo = DQN(**config["algo"])  # Run with defaults.
    agent = AtariDqnAgent()
    runner = MinibatchRlEval(
        algo=algo,
        agent=agent,
        sampler=sampler,
        n_steps=50e6,
        log_interval_steps=1e3,
        affinity=dict(cuda_idx=cuda_idx, workers_cpus=list(range(n_parallel))),
    )
    name = "dqn_" + game
    log_dir = "example_5"
    with logger_context(log_dir, run_ID, name, config):
        runner.train()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--game', help='Atari game', default='pong')
    parser.add_argument('--run_ID', help='run identifier (logging)', type=int, default=0)
    parser.add_argument('--cuda_idx', help='gpu to use ', type=int, default=None)
    parser.add_argument('--n_parallel', help='number of sampler workers', type=int, default=2)
    args = parser.parse_args()
    build_and_train(
        game=args.game,
        run_ID=args.run_ID,
        cuda_idx=args.cuda_idx,
        n_parallel=args.n_parallel,
    )
