
import psutil
import time
import torch

from rlpyt.runners.base import BaseRunner
from rlpyt.utils.logging import logger
from rlpyt.utils.quick_args import save__init__args
from rlpyt.utils.seed import set_seed, make_seed
from rlpyt.utils.prog_bar import ProgBarCounter


class UnsupervisedLearning(BaseRunner):

    def __init__(
            self,
            algo,
            n_updates,
            seed=None,
            affinity=None,
            log_interval_updates=1e3,
            snapshot_gap_intervals=None,  # units: log_intervals
            ):
        n_updates = int(n_updates)
        affinity = dict() if affinity is None else affinity
        save__init__args(locals())

    def startup(self):
        p = psutil.Process()
        try:
            if (self.affinity.get("master_cpus", None) is not None and
                    self.affinity.get("set_affinity", True)):
                p.cpu_affinity(self.affinity["master_cpus"])
            cpu_affin = p.cpu_affinity()
        except AttributeError:
            cpu_affin = "UNAVAILABLE MacOS"
        logger.log(f"Runner {getattr(self, 'rank', '')} master CPU affinity: "
            f"{cpu_affin}.")
        if self.affinity.get("master_torch_threads", None) is not None:
            torch.set_num_threads(self.affinity["master_torch_threads"])
        logger.log(f"Runner {getattr(self, 'rank', '')} master Torch threads: "
            f"{torch.get_num_threads()}.")
        if self.seed is None:
            self.seed = make_seed()
        set_seed(self.seed)
        # self.rank = rank = getattr(self, "rank", 0)
        # self.world_size = world_size = getattr(self, "world_size", 1)
        self.algo.initialize(
            n_updates=self.n_updates,
            cuda_idx=self.affinity.get("cuda_idx", None),
        )
        self.initialize_logging()

    def initialize_logging(self):
        self._opt_infos = {k: list() for k in self.algo.opt_info_fields}
        self._start_time = self._last_time = time.time()
        self._cum_time = 0.
        if self.snapshot_gap_intervals is not None:
            logger.set_snapshot_gap(
                self.snapshot_gap_intervals * self.log_interval_updates)
        self.pbar = ProgBarCounter(self.log_interval_updates)

    def shutdown(self):
        logger.log("Pretraining complete.")
        self.pbar.stop()

    def get_itr_snapshot(self, itr):
        return dict(
            itr=itr,
            algo_state_dict=self.algo.state_dict(),
        )

    def save_itr_snapshot(self, itr):
        """
        Calls the logger to save training checkpoint/snapshot (logger itself
        may or may not save, depending on mode selected).
        """
        logger.log("saving snapshot...")
        params = self.get_itr_snapshot(itr)
        logger.save_itr_params(itr, params)
        logger.log("saved")

    def store_diagnostics(self, itr, opt_info):
        for k, v in self._opt_infos.items():
            new_v = getattr(opt_info, k, [])
            v.extend(new_v if isinstance(new_v, list) else [new_v])
        self.pbar.update((itr + 1) % self.log_interval_updates)

    def log_diagnostics(self, itr, val_info, *args, **kwargs):
        self.save_itr_snapshot(itr)
        new_time = time.time()
        self._cum_time = new_time - self._start_time
        epochs = itr * self.algo.batch_size / (
            self.algo.replay_buffer.size * (1 - self.algo.validation_split)) 
        logger.record_tabular("Iteration", itr)
        logger.record_tabular("Epochs", epochs)
        logger.record_tabular("CumTime (s)", self._cum_time)
        logger.record_tabular("UpdatesPerSecond", itr / self._cum_time)
        if self._opt_infos:
            for k, v in self._opt_infos.items():
                logger.record_tabular_misc_stat(k, v)
        for k, v in zip(val_info._fields, val_info):
            logger.record_tabular_misc_stat("val_" + k, v)
        self._opt_infos = {k: list() for k in self._opt_infos}  # (reset)
        logger.dump_tabular(with_prefix=False)
        if itr < self.n_updates - 1:
            logger.log(f"Optimizing over {self.log_interval_updates} iterations.")
            self.pbar = ProgBarCounter(self.log_interval_updates)

    def train(self):
        self.startup()
        self.algo.train()
        for itr in range(self.n_updates):
            logger.set_iteration(itr)
            with logger.prefix(f"itr #{itr} "):
                opt_info = self.algo.optimize(itr)  # perform one update
                self.store_diagnostics(itr, opt_info)
                if (itr + 1) % self.log_interval_updates == 0:
                    self.algo.eval()
                    val_info = self.algo.validation(itr)
                    self.log_diagnostics(itr, val_info)
                    self.algo.train()
        self.shutdown()
