
"""
Runs multiple instances of the Atari environment and optimizes using A2C
algorithm. Can choose between configurations for use of CPU/GPU for sampling
(serial or parallel) and optimization (serial).

"""
from rlpyt.samplers.serial_sampler import SerialSampler
from rlpyt.samplers.cpu.parallel_sampler import CpuParallelSampler
from rlpyt.samplers.gpu.parallel_sampler import GpuParallelSampler
from rlpyt.samplers.cpu.collectors import ResetCollector as CpuResetCollector
from rlpyt.samplers.gpu.collectors import ResetCollector as GpuResetCollector
from rlpyt.envs.atari.atari_env import AtariEnv
from rlpyt.algos.pg.a2c import A2C
from rlpyt.agents.pg.atari import AtariFfAgent
from rlpyt.runners.minibatch_rl import MinibatchRl
from rlpyt.utils.logging.context import logger_context


def build_and_train(game="pong", run_ID=0, cuda_idx=None, sample_mode="serial", n_parallel=2):
    affinity = dict(cuda_idx=cuda_idx, workers_cpus=list(range(n_parallel)))
    gpu_cpu = "CPU" if cuda_idx is None else f"GPU {cuda_idx}"
    if sample_mode == "serial":
        Sampler = SerialSampler  # (Ignores workers_cpus.)
        Collector = CpuResetCollector
        print(f"Using serial sampler, {gpu_cpu} for sampling and optimizing.")
    elif sample_mode == "cpu":
        Sampler = CpuParallelSampler
        Collector = CpuResetCollector
        print(f"Using CPU parallel sampler (agent in workers), {gpu_cpu} for optimizing.")
    elif sample_mode == "gpu":
        Sampler = GpuParallelSampler
        Collector = GpuResetCollector
        print(f"Using GPU parallel sampler (agent in master), {gpu_cpu} for sampling and optimizing.")

    sampler = Sampler(
        EnvCls=AtariEnv,
        env_kwargs=dict(game=game),
        CollectorCls=Collector,
        batch_T=5,  # 5 time-steps per sampler iteration.
        batch_B=16,  # 16 parallel environments.
        max_decorrelation_steps=400,
    )
    algo = A2C()  # Run with defaults.
    agent = AtariFfAgent()
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
    log_dir = "example_3"
    with logger_context(log_dir, run_ID, name, config):
        runner.train()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--game', help='Atari game', default='pong')
    parser.add_argument('--run_ID', help='run identifier (logging)', type=int, default=0)
    parser.add_argument('--cuda_idx', help='gpu to use ', type=int, default=None)
    parser.add_argument('--sample_mode', help='serial or parallel sampling',
        type=str, default='serial', choices=['serial', 'cpu', 'gpu'])
    parser.add_argument('--n_parallel', help='number of sampler workers', type=int, default=2)
    args = parser.parse_args()
    build_and_train(
        game=args.game,
        run_ID=args.run_ID,
        cuda_idx=args.cuda_idx,
        sample_mode=args.sample_mode,
        n_parallel=args.n_parallel,
    )
