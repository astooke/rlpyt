
import psutil
import time

from rlpyt.utils.quick_args import save__init__args
from rlpyt.utils.seed import set_seed, make_seed
from rlpyt.utils.logging import logger
from rlpyt.utils.prog_bar import ProgBarCounter

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
        save__init__args(locals())

    def startup(self):
        p = psutil.Process()
        p.cpu_affinity(self.affinity.get("master_cpus", p.cpu_affinity()))
        logger.log(f"Set master cpu affinity: {p.cpu_affinity()}.")
        if self.seed is None:
            self.seed = make_seed()
        set_seed(self.seed)
        self.sampler.initialize(
            agent=self.agent,  # Agent gets intialized in sampler.
            affinity=self.affinity,
            seed=self.seed + 1,
            bootstrap_value=getattr(self.algo, "bootstrap_value", False),
            traj_info_kwargs=self.get_traj_info_kwargs(),
        )
        n_itr = self.get_n_itr(self.sampler.batch_spec.size)
        self.agent.initialize_cuda(self.affinity.get("cuda_idx", None))
        self.algo.initialize(self.agent, n_itr, self.sampler.mid_batch_reset)
        self.initialize_logging()
        return n_itr

    def get_traj_info_kwargs(self):
        return dict(discount=getattr(self.algo, "discount", 1))

    def get_n_itr(self, batch_size):
        n_itr = (self.n_steps + self.log_interval_steps) // batch_size + 1
        self.log_interval_itrs = max(self.log_interval_steps // batch_size, 1)
        self.n_itr = n_itr
        return n_itr

    def initialize_logging(self):
        self._opt_infos = {k: list() for k in self.algo.opt_info_fields}
        self._start_time = self._last_time = time.time()
        self.pbar = ProgBarCounter(self.log_interval_itrs)

    def shutdown(self):
        logger.log("Training complete.")
        self.pbar.stop()
        self.sampler.shutdown()

    def get_itr_snapshot(self, itr):
        return dict(
            itr=itr,
            cum_samples=itr * self.sampler.batch_spec.size,
            model_state_dict=self.agent.model.state_dict(),
            optimizer_state_dict=self.algo.optimizer.state_dict(),
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
