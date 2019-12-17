
import multiprocessing as mp
import time
import torch.distributed

from rlpyt.runners.minibatch_rl import MinibatchRl, MinibatchRlEval
from rlpyt.utils.seed import make_seed
from rlpyt.utils.collections import AttrDict
from rlpyt.utils.quick_args import save__init__args
from rlpyt.utils.synchronize import drain_queue, find_port


###############################################################################
# Master
###############################################################################


class SyncRlMixin:
    """
    Mixin class to extend runner functionality to multi-GPU case.  Creates a
    full replica of the sampler-algorithm-agent stack in a separate Python
    process for each GPU.  Initializes ``torch.distributed`` to support
    data-parallel training of the agent.  The main communication point among
    processes is to all-reduce gradients during backpropagation, which is
    handled implicitly within PyTorch.  There is one agent, with the same
    parameters copied in all processes.  No data samples are communicated in
    the implemented runners.

    On GPU, uses the `NCCL` backend to communicate directly among GPUs. Can also
    be used without GPU, as multi-CPU (MPI-like, but using the `gloo` backend).

    The parallelism in the sampler is independent from the parallelism
    here--each process will initialize its own sampler, and any one can be
    used (serial, cpu-parallel, gpu-parallel).

    The name "Sync" refers to the fact that the sampler and algorithm still
    operate synchronously within each process (i.e. they alternate, running
    one at a time).  

    Note:
       Weak scaling is implemented for batch sizes.  The batch size input
       argument to the sampler and to the algorithm classes are used in each
       process, so the actual batch sizes are `(world_size * batch_size)`.
       The world size is readily available from ``torch.distributed``, so can
       change this if desired.

    Note: 
       The ``affinities`` input is expected to be a list, with a seprate
       affinity dict for each process. The number of processes is taken from
       the length of the affinities list.
    """

    def startup(self):
        self.launch_workers()
        n_itr = super().startup()
        self.par.barrier.wait()
        self._start_time = self._last_time = time.time()  # (Overwrite)
        return n_itr

    def launch_workers(self):
        """
        As part of startup, fork a separate Python process for each additional
        GPU; the master process runs on the first GPU.  Initialize
        ``torch.distributed`` so the ``DistributedDataParallel`` wrapper can
        work--also makes ``torch.distributed`` avaiable for other
        communication.
        """
        self.affinities = self.affinity
        self.affinity = self.affinities[0]
        self.world_size = world_size = len(self.affinities)
        self.rank = rank = 0
        self.par = par = self.build_par_objs(world_size)
        if self.seed is None:
            self.seed = make_seed()
        port = find_port(offset=self.affinity.get("master_cpus", [0])[0])
        backend = "gloo" if self.affinity.get("cuda_idx", None) is None else "nccl"
        workers_kwargs = [dict(
            algo=self.algo,
            agent=self.agent,
            sampler=self.sampler,
            n_steps=self.n_steps,
            seed=self.seed + 100 * rank,
            affinity=self.affinities[rank],
            log_interval_steps=self.log_interval_steps,
            rank=rank,
            world_size=world_size,
            port=port,
            backend=backend,
            par=par,
            )
            for rank in range(1, world_size)]
        workers = [self.WorkerCls(**w_kwargs) for w_kwargs in workers_kwargs]
        self.workers = [mp.Process(target=w.train, args=()) for w in workers]
        for w in self.workers:
            w.start()
        torch.distributed.init_process_group(
            backend=backend,
            rank=rank,
            world_size=world_size,
            init_method=f"tcp://127.0.0.1:{port}",
        )

    def build_par_objs(self, world_size):
        barrier = mp.Barrier(world_size)
        traj_infos_queue = mp.Queue()
        par = AttrDict(
            barrier=barrier,
            traj_infos_queue=traj_infos_queue,
        )
        return par


class SyncRl(SyncRlMixin, MinibatchRl):
    """
    Multi-process RL with online agent performance tracking.  Trajectory info is
    collected from all processes and is included in logging.
    """

    @property
    def WorkerCls(self):
        return SyncWorker

    def store_diagnostics(self, itr, traj_infos, opt_info):
        traj_infos.extend(drain_queue(self.par.traj_infos_queue))
        super().store_diagnostics(itr, traj_infos, opt_info)


class SyncRlEval(SyncRlMixin, MinibatchRlEval):
    """
    Multi-process RL with offline agent performance evaluation.  Only the
    master process runs agent evaluation.
    """

    @property
    def WorkerCls(self):
        return SyncWorkerEval

    def log_diagnostics(self, *args, **kwargs):
        super().log_diagnostics(*args, **kwargs)
        self.par.barrier.wait()


###############################################################################
# Worker
###############################################################################


class SyncWorkerMixin:

    def __init__(
            self,
            algo,
            agent,
            sampler,
            n_steps,
            seed,
            affinity,
            log_interval_steps,
            rank,
            world_size,
            port,
            backend,
            par,
            ):
        save__init__args(locals())

    def startup(self):
        torch.distributed.init_process_group(
            backend=self.backend,
            rank=self.rank,
            world_size=self.world_size,
            init_method=f"tcp://127.0.0.1:{self.port}",
        )
        n_itr = super().startup()
        self.par.barrier.wait()
        return n_itr

    def initialize_logging(self):
        pass  # Don't log in workers.

    def shutdown(self):
        self.sampler.shutdown()


class SyncWorker(SyncWorkerMixin, MinibatchRl):

    def store_diagnostics(self, itr, traj_infos, opt_info):
        for traj_info in traj_infos:
            self.par.traj_infos_queue.put(traj_info)
        # Leave worker opt_info un-recorded.

    def log_diagnostics(self, *args, **kwargs):
        pass


class SyncWorkerEval(SyncWorkerMixin, MinibatchRlEval):

    def store_diagnostics(self, *args, **kwargs):
        pass

    def log_diagnostics(self, *args, **kwargs):
        self.par.barrier.wait()

    def evaluate_agent(self, *args, **kwargs):
        return None, None
