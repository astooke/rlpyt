
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

    def startup(self):
        self.launch_workers()
        n_itr = super().startup()
        self.par.barrier.wait()
        self._start_time = self._last_time = time.time()  # (Overwrite)
        return n_itr

    def launch_workers(self):
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

    @property
    def WorkerCls(self):
        return SyncWorker

    def store_diagnostics(self, itr, traj_infos, opt_info):
        traj_infos.extend(drain_queue(self.par.traj_infos_queue))
        super().store_diagnostics(itr, traj_infos, opt_info)


class SyncRlEval(SyncRlMixin, MinibatchRlEval):

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
