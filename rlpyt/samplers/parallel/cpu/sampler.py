
import multiprocessing as mp
import time


from rlpyt.samplers.base import BaseSampler
from rlpyt.samplers.parallel.base import ParallelSamplerBase
from rlpyt.samplers.utils import build_samples_buffer, build_par_objs
from rlpyt.samplers.parallel_worker import sampling_process
from rlpyt.samplers.cpu.collectors import CpuResetCollector, EvalCollector
from rlpyt.utils.logging import logger
from rlpyt.utils.synchronize import drain_queue


EVAL_TRAJ_CHECK = 1  # Seconds.


class CpuSampler(ParallelSamplerBase):

    def __init__(self, *args, CollectorCls=CpuResetCollector,
            eval_CollectorCls=CpuEvalCollector, **kwargs):
        # e.g. or use CpuWaitResetCollector, etc...
        super().__init__(*args, CollectorCls=CollectorCls,
            eval_CollectorCls=eval_CollectorCls, **kwargs)

    def obtain_samples(self, itr):
        self.agent.sync_shared_memory()  # New weights in workers, if needed.
        return super().obtain_samples(itr)

    def evaluate_agent(self, itr):
        self.agent.sync_shared_memory()
        return super().evaluate_agent(itr)
