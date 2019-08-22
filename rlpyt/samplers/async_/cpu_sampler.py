
import time
import psutil
import torch

from rlpyt.samplers.async_.base import AsyncParallelSamplerMixin
from rlpyt.samplers.parallel.base import ParallelSamplerBase
from rlpyt.samplers.async_.collectors import DbCpuResetCollector
from rlpyt.samplers.parallel.cpu.collectors import CpuEvalCollector
from rlpyt.utils.logging import logger
from rlpyt.utils.synchronize import drain_queue


EVAL_TRAJ_CHECK = 0.1  # Seconds.


class AsyncCpuSampler(AsyncParallelSamplerMixin, ParallelSamplerBase):

    def __init__(self, *args, CollectorCls=DbCpuResetCollector,
            eval_CollectorCls=CpuEvalCollector, **kwargs):
        super().__init__(*args, CollectorCls=CollectorCls,
            eval_CollectorCls=eval_CollectorCls, **kwargs)

    ###########################################################################
    # Sampler runner methods (forked).
    ###########################################################################

    def initialize(self, affinity):
        p = psutil.Process()
        if affinity.get("set_affinity", True):
            p.cpu_affinity(affinity["master_cpus"])
        torch.set_num_threads(1)  # Needed to prevent MKL hang :( .
        self.agent.async_cpu(share_memory=True)
        super().initialize(
            agent=self.agent,
            affinity=affinity,
            seed=self.seed,
            bootstrap_value=None,  # Don't need here.
            traj_info_kwargs=None,  # Already done.
            world_size=1,
            rank=0,
        )

    def obtain_samples(self, itr, db_idx):
        self.agent.recv_shared_memory()
        return super().obtain_samples(itr, db_idx)

    def evaluate_agent(self, itr):
        self.agent.recv_shared_memory()
        return super().evaluate_agent(itr)
