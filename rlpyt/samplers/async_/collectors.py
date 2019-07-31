
from rlpyt.samplers.gpu.collectors import ResetCollector as GpuRC
from rlpyt.samplers.gpu.collectors import WaitResetCollector as GpuWRC
from rlpyt.samplers.cpu.collectors import ResetCollector as CpuRC
from rlpyt.samplers.cpu.collectors import WaitResetCollector as CpuWRC


class DoubleBufferCollectorMixin(object):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.double_buffer = self.samples_np
        self.samples_np = self.double_buffer[0]

    def collect_batch(self, *args, **kwargs):
        self.samples_np = self.double_buffer[self.sync.db_idx.value]
        return super().collect_batch(*args, **kwargs)


class DbGpuResetCollector(DoubleBufferCollectorMixin, GpuRC):
    pass


class DbGpuWaitResetCollector(DoubleBufferCollectorMixin, GpuWRC):
    pass


class DbCpuResetCollector(DoubleBufferCollectorMixin, CpuRC):
    pass


class DbCpuWaitResetCollector(DoubleBufferCollectorMixin, CpuWRC):
    pass
