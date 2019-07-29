
import multiprocessing as mp
import queue


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


def drain_queue(queue_obj):
    contents = list()
    while True:
        try:
            contents.append(queue_obj.get(block=False))
        except queue.Empty:
            return contents


def find_port(offset):
    # Find a unique open port, to stack multiple multi-GPU runs per machine.
    import torch.distributed
    assert offset < 100
    for port in range(29500 + offset, 65000, 100):
        try:
            store = torch.distributed.TCPStore("127.0.0.1", port, 1, True)
            break
        except RuntimeError:
            pass  # Port taken.
    del store  # Before fork (small time gap; could be re-taken, hence offset).
    return port
