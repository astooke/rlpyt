
from rlpyt.samplers.gpu.base import GpuParallelSamplerBase
from rlpyt.samplers.gpu.action_server import ActionServer


class GpuParallelSampler(ActionServer, GpuParallelSamplerBase):
    pass
