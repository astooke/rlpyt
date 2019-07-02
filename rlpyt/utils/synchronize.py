
import multiprocessing as mp


class RWLock(object):
    """Multiple simultaneous readers, one writer."""

    def __init__(self):
        self.write_lock = mp.Lock()
        self._read_lock = mp.Lock()
        self._read_count = mp.RawValue("i")

    def __enter__(self):
        self.acquire_read()

    def __exit__(self, *args):
        self.release_read()

    def acquire_write(self):
        self.write_lock.acquire()

    def release_write(self):
        self.write_lock.release()

    def acquire_read(self):
        with self._read_lock:
            self._read_count.value += 1
            if self._read_count.value == 1:
                self.write_lock.acquire()

    def release_read(self):
        with self._read_lock:
            self._read_count.value -= 1
            if self._read_count.value == 0:
                self.write_lock.release()
