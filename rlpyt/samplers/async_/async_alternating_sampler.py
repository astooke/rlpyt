
import numpy as np 

from rlpyt.samplers.async_.async_gpu_sampler import AsyncGpuSamplerBase
from rlpyt.samplers.async_.action_server import (AsyncAlternatingActionServer,
    AsyncNoOverlapAlternatingActionServer)
from rlpyt.util.logging import logger
from rlpyt.utils.synchronize import drain_queue


class AsyncAlternatingSamplerBase(AsyncGpuSamplerBase):

    alternating = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.batch_spec.B % 2 == 0, "Need even number for sampler batch_B."

    def async_initialize(self, agent, *args, **kwargs):
        if agent.recurrent and not agent.alternating:
            raise TypeError("If agent is recurrent, must be 'alternating' to use here.")
        elif not agent.recurrent:
            agent.alternating = True  # FF agent doesn't need special class, but tell it so.
        return super().async_initialize(agent, *args, **kwargs)


class AsyncAlternatingSampler(AsyncAlternatingActionServer, 
        AsyncAlternatingSamplerBase):
    pass


class AsyncNoOverlapAlternatingSampler(AsyncNoOverlapAlternatingActionServer,
        AsyncAlternatingSamplerBase):
    pass
