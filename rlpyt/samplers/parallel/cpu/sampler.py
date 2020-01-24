
import multiprocessing as mp
import time


from rlpyt.samplers.parallel.base import ParallelSamplerBase
from rlpyt.samplers.parallel.cpu.collectors import (CpuResetCollector,
    CpuEvalCollector)


class CpuSampler(ParallelSamplerBase):
    """Parallel sampler for using the CPU resource of each worker to
    compute agent forward passes; for use with CPU-based collectors.
    """

    def __init__(self, *args, CollectorCls=CpuResetCollector,
            eval_CollectorCls=CpuEvalCollector, **kwargs):
        # e.g. or use CpuWaitResetCollector, etc...
        super().__init__(*args, CollectorCls=CollectorCls,
            eval_CollectorCls=eval_CollectorCls, **kwargs)

    def obtain_samples(self, itr):
        """First, have the agent sync shared memory; in case training uses a
        GPU, the agent needs to copy its (new) GPU model parameters to the
        shared-memory CPU model which all the workers use.  Then call super
        class's method.
        """
        self.agent.sync_shared_memory()  # New weights in workers, if needed.
        return super().obtain_samples(itr)

    def evaluate_agent(self, itr):
        """Like in ``obtain_samples()``, first sync agent shared memory."""
        self.agent.sync_shared_memory()
        return super().evaluate_agent(itr)
