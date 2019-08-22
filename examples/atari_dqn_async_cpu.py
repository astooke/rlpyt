
"""
DQN in async mode with CPU parallel sampler.
"""


from rlpyt.utils.launching.affinity import make_affinity
from rlpyt.samplers.async_.cpu_sampler import AsyncCpuSampler
from rlpyt.envs.atari.atari_env import AtariEnv
from rlpyt.algos.dqn.dqn import DQN
from rlpyt.agents.dqn.atari.atari_dqn_agent import AtariDqnAgent
from rlpyt.runners.async_rl import AsyncRlEval
from rlpyt.utils.logging.context import logger_context


def build_and_train(game="pong", run_ID=0):
    # Change these inputs to match local machine and desired parallelism.
    affinity = make_affinity(
        run_slot=0,
        n_cpu_core=3,  # Use 16 cores across all experiments.
        n_gpu=1,  # Use 8 gpus across all experiments.
        sample_gpu_per_run=0,
        async_sample=True,
        # hyperthread_offset=24,  # If machine has 24 cores.
        # n_socket=2,  # Presume CPU socket affinity to lower/upper half GPUs.
        # gpu_per_run=2,  # How many GPUs to parallelize one run across.
        # cpu_per_run=1,
    )

    sampler = AsyncCpuSampler(
        EnvCls=AtariEnv,
        env_kwargs=dict(game=game),
        batch_T=5,
        batch_B=8,
        max_decorrelation_steps=100,
        eval_env_kwargs=dict(game=game),
        eval_n_envs=2,
        eval_max_steps=int(10e3),
        eval_max_trajectories=4,
    )
    algo = DQN(
        replay_ratio=8,
        min_steps_learn=1e4,
        replay_size=int(1e5)
    )
    agent = AtariDqnAgent()
    runner = AsyncRlEval(
        algo=algo,
        agent=agent,
        sampler=sampler,
        n_steps=2e6,
        log_interval_steps=1e4,
        affinity=affinity,
    )
    config = dict(game=game)
    name = "async_dqn_" + game
    log_dir = "async_dqn"
    with logger_context(log_dir, run_ID, name, config):
        runner.train()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--game', help='Atari game', default='pong')
    parser.add_argument('--run_ID', help='run identifier (logging)', type=int, default=0)
    args = parser.parse_args()
    build_and_train(
        game=args.game,
        run_ID=args.run_ID,
    )
