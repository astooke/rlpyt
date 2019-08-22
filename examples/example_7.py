
"""
Runs one experiment using multiple GPUs. In the SyncRl runner, the entire
template of optimizer and sampler GPU and CPUs is replicated across the
machine, with a separate python process for each GPU. So each parallel runner
gathers its own samples for optimization.  The models are parallelized using
Torch's DistributedDataParallel, which all-reduces every gradient computed,
pipelined with backpropagation; all GPUs maintain identical copies of the
model throughout.  The same technique can be applied to any algorithm, PG,
QPG, DQN for multi-GPU training synchronous with sampling.

Currently, the batch size specified to the sampler/algo is used on each process,
so total batch size grows with the number of parallel runners.

Try different affinity inputs to see where the jobs run on the machine.

"""
from rlpyt.utils.launching.affinity import make_affinity
from rlpyt.samplers.parallel.gpu.sampler import GpuParallelSampler
from rlpyt.samplers.parallel.gpu.collectors import GpuWaitResetCollector
from rlpyt.envs.atari.atari_env import AtariEnv
from rlpyt.algos.pg.a2c import A2C
from rlpyt.agents.pg.atari import AtariFfAgent
from rlpyt.runners.sync_rl import SyncRl
from rlpyt.utils.logging.context import logger_context


def build_and_train(game="pong", run_ID=0):
    # Seems like we should be able to skip the intermediate step of the code,
    # but so far have just always run that way.
    # Change these inputs to match local machine and desired parallelism.
    affinity = make_affinity(
        run_slot=0,
        n_cpu_cores=16,  # Use 16 cores across all experiments.
        n_gpu=8,  # Use 8 gpus across all experiments.
        hyperthread_offset=24,  # If machine has 24 cores.
        n_socket=2,  # Presume CPU socket affinity to lower/upper half GPUs.
        gpu_per_run=2,  # How many GPUs to parallelize one run across.
        # cpu_per_run=1,
    )

    sampler = GpuParallelSampler(
        EnvCls=AtariEnv,
        env_kwargs=dict(game=game),
        CollectorCls=GpuWaitResetCollector,
        batch_T=5,
        batch_B=16,
        max_decorrelation_steps=400,
    )
    algo = A2C()  # Run with defaults.
    agent = AtariFfAgent()
    runner = SyncRl(
        algo=algo,
        agent=agent,
        sampler=sampler,
        n_steps=50e6,
        log_interval_steps=1e5,
        affinity=affinity,
    )
    config = dict(game=game)
    name = "a2c_" + game
    log_dir = "example_7"
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
