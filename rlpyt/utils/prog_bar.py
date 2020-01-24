

import pyprind
from rlpyt.utils.logging import logger


class ProgBarCounter:
    """Dynamic display of progress bar in terminal, for example to mark progress
    (and estimate time to completion) of RL iterations toward the next logging
    update.  credit: `rllab`.
    """

    def __init__(self, total_count):
        self.total_count = total_count
        self.max_progress = 1000000
        self.cur_progress = 0
        self.cur_count = 0
        if not logger.get_log_tabular_only():
            self.pbar = pyprind.ProgBar(self.max_progress)
        else:
            self.pbar = None

    def update(self, current_count):
        if not logger.get_log_tabular_only():
            self.cur_count = current_count
            new_progress = self.cur_count * self.max_progress / self.total_count
            if new_progress < self.max_progress:
                self.pbar.update(new_progress - self.cur_progress)
            self.cur_progress = new_progress

    def stop(self):
        if self.pbar is not None and self.pbar.active:
                self.pbar.stop()
