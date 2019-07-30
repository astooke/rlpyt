
import multiprocessing as mp
import ctypes

from rlpyt.utils.synchronize import RWLock


class AsyncReplayBufferMixin(object):

    def __init__(self, *args, **kwargs):
        kwargs.pop("shared_memory")
        super().__init__(*args, shared_memory=True, **kwargs)
        self.async_t = mp.RawValue("l")  # Type c_long.
        self.rw_lock = RWLock()
        self._async_buffer_full = mp.RawValue(ctypes.c_bool, False)

    def append_samples(self, *args, **kwargs):
        with self.rw_lock.write_lock:
            self.t = self.async_t.value
            ret = super().append_samples(*args, **kwargs)
            self.async_t.value = self.t
            self._async_buffer_full.value = self._buffer_full
        return ret

    def sample_batch(self, *args, **kwargs):
        with self.rw_lock:
            self.t = self.async_t.value
            self._buffer_full = self._async_buffer_full.value
            return super().sample_batch(*args, **kwargs)

    def update_batch_priorities(self, *args, **kwargs):
        with self.rw_lock.write_lock:
            return super().update_batch_priorities(*args, **kwargs)
