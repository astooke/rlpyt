
import multiprocessing as mp

from rlpyt.utils.synchronize import RWLock


class AsyncReplayBufferMixin(object):

    def __init__(self, *args, **kwargs):
        kwargs.pop("shared_memory")
        super().__init__(*args, shared_memory=True, **kwargs)
        self.async_t = mp.RawValue("l")  # Type c_long.
        self.rw_lock = RWLock()

    def append_samples(self, *args, **kwargs):
        with self.rw_lock.write_lock:
            ret = super().append_samples(*args, **kwargs)
            self.async_t.value = self.t
            return ret

    def sample_batch(self, *args, **kwargs):
        with self.rw_lock:
            self.t = self.async_t.value
            return super().sample_batch(*args, **kwargs)
