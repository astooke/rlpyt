
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
    """Parallel sampler for agent action-selection on CPU, to use in
    asynchronous runner.  The master (training) process will have forked
    the main sampler process, which here will fork sampler workers from
    itself, and otherwise will run similarly to the ``CpuSampler``.
    """

    def __init__(self, *args, CollectorCls=DbCpuResetCollector,
            eval_CollectorCls=CpuEvalCollector, **kwargs):
        super().__init__(*args, CollectorCls=CollectorCls,
            eval_CollectorCls=eval_CollectorCls, **kwargs)

    ###########################################################################
    # Sampler runner methods (forked).
    ###########################################################################

    def initialize(self, affinity):
        """
        Runs inside the main sampler process.  Sets process hardware affinity
        and calls the ``agent.async_cpu()`` initialization.  Then proceeds with
        usual parallel sampler initialization.
        """
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
        """Calls the agent to retrieve new parameter values from the training
        process, then proceeds with base async parallel method.
        """
        self.agent.recv_shared_memory()
        return super().obtain_samples(itr, db_idx)

    def evaluate_agent(self, itr):
        """Calls the agent to retrieve new parameter values from the training
        process, then proceeds with base async parallel method.
        """
        self.agent.recv_shared_memory()
        return super().evaluate_agent(itr)
