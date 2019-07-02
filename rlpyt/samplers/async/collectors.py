
from rlpyt.samplers.gpu.collectors import ResetCollector, WaitResetCollector


class DoubleBufferCollectorMixin(object):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.double_buffer = self.samples_np
        self.j = 0
        self.samples_np = self.double_buffer[self.j]

    def collect_batch(self, *args, **kwargs):
        self.samples_np = self.double_buffer[self.j]
        super().collect_batch(*args, **kwargs)
        self.j ^= 1


class DbResetCollector(DoubleBufferCollectorMixin, ResetCollector):
    pass


class DbWaitResetCollector(DoubleBufferCollectorMixin, WaitResetCollector):
    pass
