
import psutil
import time
import torch
import math

from rlpyt.utils.quick_args import save__init__args
from rlpyt.utils.seed import set_seed, make_seed
from rlpyt.utils.logging import logger
from rlpyt.runners.base import BaseRunner


class MinibatchRlBase(BaseRunner):

    def __init__(
            self,
            algo,
            agent,
            sampler,
            n_steps,
            seed=None,
            affinity=None,
            log_interval_steps=1e5,
            ):
        n_steps = int(n_steps)
        log_interval_steps = int(log_interval_steps)
        affinity = dict() if affinity is None else affinity
        save__init__args(locals())

    def startup(self):
        p = psutil.Process()
        try:
            if "master_cpus" in self.affinity:
                p.cpu_affinity(self.affinity["master_cpus"])
            cpu_affin = p.cpu_affinity()
        except AttributeError:
            cpu_affin = "UNAVAILABLE MacOS"
        logger.log(f"Runner {getattr(self, 'rank', '')} master CPU affinity: "
            f"{cpu_affin}.")
        if "master_torch_threads" in self.affinity:
            torch.set_num_threads(self.affinity["master_torch_threads"])
        logger.log(f"Runner {getattr(self, 'rank', '')} master Torch threads: "
            f"{torch.get_num_threads()}.")
        if self.seed is None:
            self.seed = make_seed()
        set_seed(self.seed)
        examples = self.sampler.initialize(
            agent=self.agent,  # Agent gets intialized in sampler.
            affinity=self.affinity,
            seed=self.seed + 1,
            bootstrap_value=getattr(self.algo, "bootstrap_value", False),
            traj_info_kwargs=self.get_traj_info_kwargs(),
        )
        n_runners = getattr(self, "n_runners", 1)
        self.itr_batch_size = self.sampler.batch_spec.size * n_runners
        n_itr = self.get_n_itr()
        self.agent.initialize_cuda(
            cuda_idx=self.affinity.get("cuda_idx", None),
            ddp=n_runners > 1,  # Multi-GPU training (and maybe sampling).
        )
        self.algo.initialize(
            agent=self.agent,
            n_itr=n_itr,
            batch_spec=self.sampler.batch_spec,
            mid_batch_reset=self.sampler.mid_batch_reset,
            examples=examples
        )
        self.initialize_logging()
        return n_itr

    def get_traj_info_kwargs(self):
        return dict(discount=getattr(self.algo, "discount", 1))

    def get_n_itr(self):
        log_interval_itrs = max(self.log_interval_steps //
            self.itr_batch_size, 1)
        n_itr = math.ceil(self.n_steps / self.log_interval_steps) * log_interval_itrs
        self.log_interval_itrs = log_interval_itrs
        self.n_itr = n_itr
        logger.log(f"Running {n_itr} iterations of minibatch RL.")
        return n_itr

    def initialize_logging(self):
        self._opt_infos = {k: list() for k in self.algo.opt_info_fields}
        self._start_time = self._last_time = time.time()

    def shutdown(self):
        logger.log("Training complete.")
        self.pbar.stop()
        self.sampler.shutdown()

    def get_itr_snapshot(self, itr):
        return dict(
            itr=itr,
            cum_steps=itr * self.itr_batch_size,
            agent_state_dict=self.agent.state_dict(),
            optimizer_state_dict=self.algo.optim_state_dict(),
        )

    def save_itr_snapshot(self, itr):
        logger.log("saving snapshot...")
        params = self.get_itr_snapshot(itr)
        logger.save_itr_params(itr, params)
        logger.log("saved")

    def _log_infos(self, traj_infos=None):
        if traj_infos is None:
            traj_infos = self._traj_infos
        if traj_infos:
            for k in traj_infos[0]:
                if not k.startswith("_"):
                    logger.record_tabular_misc_stat(k,
                        [info[k] for info in traj_infos])

        if self._opt_infos:
            for k, v in self._opt_infos.items():
                logger.record_tabular_misc_stat(k, v)
        self._opt_infos = {k: list() for k in self._opt_infos}  # (reset)
