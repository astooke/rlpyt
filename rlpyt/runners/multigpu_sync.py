
import multiprocessing as mp
import time
import torch.distributed

from rlpyt.runners.minibatch_rl import MinibatchRl
from rlpyt.runners.minibatch_rl_eval import MinibatchRlEval
from rlpyt.utils.seed import make_seed
from rlpyt.utils.collections import AttrDict
from rlpyt.utils.quick_args import save__init__args


###############################################################################
# Master
###############################################################################


class MultiGpuRlMixin(object):

    def startup(self):
        self.launch_workers()
        n_itr = super().startup()
        self.par.barrier.wait()
        self._start_time = self._last_time = time.time()  # (Overwrite)
        return n_itr

    def launch_workers(self):
        self.affinities = self.affinity
        self.affinity = self.affinities[0]
        self.n_runners = n_runners = len(self.affinities)
        self.rank = rank = 0
        self.par = par = self.build_par_objs(n_runners)
        if self.seed is None:
            self.seed = make_seed()
        port = find_port(offset=self.affinity.get("run_slot", 0))
        workers_kwargs = [dict(
            algo=self.algo,
            agent=self.agent,
            sampler=self.sampler,
            n_steps=self.n_steps,
            seed=self.seed + 100 * rank,
            affinity=self.affinities[rank],
            log_interval_steps=self.log_interval_steps,
            rank=rank,
            n_runners=n_runners,
            port=port,
            par=par,
            )
            for rank in range(1, n_runners)]
        workers = [self.WorkerCls(**w_kwargs) for w_kwargs in workers_kwargs]
        self.workers = [mp.Process(target=w.train, args=()) for w in workers]
        for w in self.workers:
            w.start()
        torch.distributed.init_process_group(
            backend="nccl",
            rank=rank,
            world_size=n_runners,
            init_method=f"tcp://127.0.0.1:{port}",
        )

    def build_par_objs(self, n_runners):
        barrier = mp.Barrier(n_runners)
        traj_infos_queue = mp.Queue()
        mgr = mp.Manager()
        mgr_dict = mgr.dict()  # For any other comms.
        par = AttrDict(
            barrier=barrier,
            traj_infos_queue=traj_infos_queue,
            dict=mgr_dict,
        )
        return par


class MultiGpuRl(MultiGpuRlMixin, MinibatchRl):

    @property
    def WorkerCls(self):
        return MultiGpuWorker

    def store_diagnostics(self, itr, traj_infos, opt_info):
        while self.par.traj_infos_queue.qsize():
            traj_infos.append(self.par.traj_infos_queue.get())
        super().store_diagnostics(itr, traj_infos, opt_info)


class MultiGpuRlEval(MultiGpuRlMixin, MinibatchRlEval):

    @property
    def WorkerCls(self):
        return MultiGpuWorkerEval

    def log_diagnostics(self, *args, **kwargs):
        super().log_diagnostics(*args, **kwargs)
        self.par.barrier.wait()


###############################################################################
# Worker
###############################################################################


class MultiGpuWorkerMixin(object):

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
            n_runners,
            port,
            par,
            ):
        save__init__args(locals())

    def startup(self):
        torch.distributed.init_process_group(
            backend="nccl",
            rank=self.rank,
            world_size=self.n_runners,
            init_method=f"tcp://127.0.0.1:{self.port}",
        )
        n_itr = super().startup()
        self.par.barrier.wait()
        return n_itr

    def initialize_logging(self):
        pass  # Don't log in workers.

    def shutdown(self):
        self.sampler.shutdown()


class MultiGpuWorker(MultiGpuWorkerMixin, MinibatchRl):

    def store_diagnostics(self, itr, traj_infos, opt_info):
        for traj_info in traj_infos:
            self.par.traj_infos_queue.put(traj_info)
        # Leave worker opt_info un-recorded.

    def log_diagnostics(self, *args, **kwargs):
        pass


class MultiGpuWorkerEval(MultiGpuWorkerMixin, MinibatchRlEval):

    def store_diagnostics(self, *args, **kwargs):
        pass

    def log_diagnostics(self, *args, **kwargs):
        self.par.barrier.wait()

    def evaluate_agent(self, *args, **kwargs):
        return None, None


# Helpers


def find_port(offset):
    # Find a unique open port, to stack multiple multi-GPU runs per machine.
    assert offset < 100
    for port in range(29500 + offset, 65000, 100):
        try:
            store = torch.distributed.TCPStore("127.0.0.1", port, 1, True)
            break
        except RuntimeError:
            pass  # Port taken.
    del store  # Before fork (small time gap; could be re-taken, hence offset).
    return port
