
from rlpyt.samplers.parallel.cpu.collectors import ResetCollector as CpuRC
from rlpyt.samplers.parallel.cpu.collectors import WaitResetCollector as CpuWRC
from rlpyt.samplers.parallel.gpu.collectors import ResetCollector as GpuRC
from rlpyt.samplers.parallel.gpu.collectors import WaitResetCollector as GpuWRC


class DoubleBufferCollectorMixin:

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.double_buffer = self.samples_np
        self.samples_np = self.double_buffer[0]

    def collect_batch(self, *args, **kwargs):
        """Swap in the called-for double buffer to record samples into."""
        self.samples_np = self.double_buffer[self.sync.db_idx.value]
        return super().collect_batch(*args, **kwargs)


class DbCpuResetCollector(DoubleBufferCollectorMixin, CpuRC):
    pass


class DbCpuWaitResetCollector(DoubleBufferCollectorMixin, CpuWRC):
    pass


class DbGpuResetCollector(DoubleBufferCollectorMixin, GpuRC):
    pass


class DbGpuWaitResetCollector(DoubleBufferCollectorMixin, GpuWRC):
    pass
