
import multiprocessing as mp
import queue


class RWLock:
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
        self.write_lock.acquire()  # or use `with rw_lock.write_lock:`.

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


def drain_queue(queue_obj, n_sentinel=0, guard_sentinel=False):
    """Empty a multiprocessing queue object, with options to protect against
    the delay between ``queue.put()`` and ``queue.get()``.  Returns a list of
    the queue contents.

    With ``n_sentinel=0``, simply call ``queue.get(block=False)`` until
    ``queue.Empty`` exception (which can still happen slightly *after* another
    process called ``queue.put()``).

    With ``n_sentinel>1``, call ``queue.get()`` until `n_sentinel` ``None``
    objects have been returned (marking that each ``put()`` process has finished).

    With ``guard_sentinel=True`` (need ``n_sentinel=0``), stops if a ``None``
    is retrieved, and puts it back into the queue, so it can do a blocking
    drain later with ``n_sentinel>1``.
    """
    contents = list()
    if n_sentinel > 0:  # Block until this many None (sentinels) received.
        sentinel_counter = 0
        while True:
            obj = queue_obj.get()
            if obj is None:
                sentinel_counter += 1
                if sentinel_counter >= n_sentinel:
                    return contents
            else:
                contents.append(obj)
    while True:  # Non-blocking, beware of delay between put() and get().
        try:
            obj = queue_obj.get(block=False)
        except queue.Empty:
            return contents
        if guard_sentinel and obj is None:
            # Restore sentinel, intend to do blocking drain later.
            queue_obj.put(None)
            return contents
        elif obj is not None:  # Ignore sentinel.
            contents.append(obj)


def find_port(offset):
    """Find a unique open port, for initializing `torch.distributed` in
    multiple separate multi-GPU runs on one machine."""
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
