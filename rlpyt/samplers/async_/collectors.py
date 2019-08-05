
from rlpyt.samplers.parallel.cpu.collectors import (CpuResetCollector,
    CpuWaitResetCollector)
from rlpyt.samplers.parallel.gpu.collectors import (GpuResetCollector,
    GpuWaitResetCollector)


class DoubleBufferCollectorMixin:

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.double_buffer = self.samples_np
        self.samples_np = self.double_buffer[0]

    def collect_batch(self, *args, **kwargs):
        """Swap in the called-for double buffer to record samples into."""
        self.samples_np = self.double_buffer[self.sync.db_idx.value]
        return super().collect_batch(*args, **kwargs)


class DbCpuResetCollector(DoubleBufferCollectorMixin, CpuResetCollector):
    pass


class DbCpuWaitResetCollector(DoubleBufferCollectorMixin, CpuWaitResetCollector):
    pass


class DbGpuResetCollector(DoubleBufferCollectorMixin, GpuResetCollector):
    pass


class DbGpuWaitResetCollector(DoubleBufferCollectorMixin, GpuWaitResetCollector):
    pass
