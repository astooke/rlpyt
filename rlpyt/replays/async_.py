
import multiprocessing as mp
import ctypes

from rlpyt.utils.synchronize import RWLock


class AsyncReplayBufferMixin:
    """Mixin class which manages the buffer (shared) memory under a read-write
    lock (multiple-reader, single-writer), for use with the asynchronous
    runner. Wraps the ``append_samples()``, ``sample_batch()``, and
    ``update_batch_priorities()`` methods. Maintains a universal buffer
    cursor, communicated asynchronously.  Supports multiple buffer-writer
    processes and multiple replay processes.
    """

    async_ = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.async_t = mp.RawValue("l")  # Type c_long.
        self.rw_lock = RWLock()
        self._async_buffer_full = mp.RawValue(ctypes.c_bool, False)

    def append_samples(self, *args, **kwargs):
        with self.rw_lock.write_lock:
            self._async_pull()  # Updates from other writers.
            ret = super().append_samples(*args, **kwargs)
            self._async_push()  # Updates to other writers + readers.
        return ret

    def sample_batch(self, *args, **kwargs):
        with self.rw_lock:  # Read lock.
            self._async_pull()  # Updates from writers.
            return super().sample_batch(*args, **kwargs)

    def update_batch_priorities(self, *args, **kwargs):
        with self.rw_lock.write_lock:
            return super().update_batch_priorities(*args, **kwargs)

    def _async_pull(self):
        self.t = self.async_t.value
        self._buffer_full = self._async_buffer_full.value

    def _async_push(self):
        self.async_t.value = self.t
        self._async_buffer_full.value = self._buffer_full
