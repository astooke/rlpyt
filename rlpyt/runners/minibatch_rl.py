
import psutil
import time
import torch
import math
from collections import deque

from rlpyt.runners.base import BaseRunner
from rlpyt.utils.quick_args import save__init__args
from rlpyt.utils.seed import set_seed, make_seed
from rlpyt.utils.logging import logger
from rlpyt.utils.prog_bar import ProgBarCounter


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
            if self.affinity.get("master_cpus", None) is not None:
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
        rank = getattr(self, "rank", 0)
        world_size = getattr(self, "world_size", 1)
        examples = self.sampler.initialize(
            agent=self.agent,  # Agent gets intialized in sampler.
            affinity=self.affinity,
            seed=self.seed + 1,
            bootstrap_value=getattr(self.algo, "bootstrap_value", False),
            traj_info_kwargs=self.get_traj_info_kwargs(),
            rank=rank,
            world_size=world_size,
        )
        self.itr_batch_size = self.sampler.batch_spec.size * world_size
        n_itr = self.get_n_itr()
        self.agent.initialize_device(
            cuda_idx=self.affinity.get("cuda_idx", None),
            ddp=world_size > 1,  # Multi-GPU training (and maybe sampling).
        )
        self.algo.initialize(
            agent=self.agent,
            n_itr=n_itr,
            batch_spec=self.sampler.batch_spec,
            mid_batch_reset=self.sampler.mid_batch_reset,
            examples=examples,
            rank=rank,
            world_size=world_size,
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

    def log_diagnostics(self, itr):
        pass

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


class MinibatchRl(MinibatchRlBase):
    """Runs RL on minibatches; tracks performance online using learning
    trajectories."""

    def __init__(self, log_traj_window=100, **kwargs):
        super().__init__(**kwargs)
        self.log_traj_window = int(log_traj_window)

    def train(self):
        n_itr = self.startup()
        for itr in range(n_itr):
            with logger.prefix(f"itr #{itr} "):
                self.agent.sample_mode(itr)  # Might not be this agent sampling.
                samples, traj_infos = self.sampler.obtain_samples(itr)
                self.agent.train_mode(itr)
                opt_info = self.algo.optimize_agent(itr, samples)
                self.store_diagnostics(itr, traj_infos, opt_info)
                if (itr + 1) % self.log_interval_itrs == 0:
                    self.log_diagnostics(itr)
        self.shutdown()

    def initialize_logging(self):
        self._traj_infos = deque(maxlen=self.log_traj_window)
        self._cum_completed_trajs = 0
        self._new_completed_trajs = 0
        logger.log(f"Optimizing over {self.log_interval_itrs} iterations.")
        super().initialize_logging()
        self.pbar = ProgBarCounter(self.log_interval_itrs)

    def store_diagnostics(self, itr, traj_infos, opt_info):
        self._cum_completed_trajs += len(traj_infos)
        self._new_completed_trajs += len(traj_infos)
        self._traj_infos.extend(traj_infos)
        for k, v in self._opt_infos.items():
            new_v = getattr(opt_info, k, [])
            v.extend(new_v if isinstance(new_v, list) else [new_v])
        self.pbar.update((itr + 1) % self.log_interval_itrs)

    def log_diagnostics(self, itr):
        self.pbar.stop()
        self.save_itr_snapshot(itr)
        new_time = time.time()
        samples_per_second = (self.log_interval_itrs *
            self.itr_batch_size) / (new_time - self._last_time)
        logger.record_tabular('Iteration', itr)
        logger.record_tabular('CumSteps', (itr + 1) * self.itr_batch_size)
        logger.record_tabular('CumTime (s)', new_time - self._start_time)
        logger.record_tabular('SamplesPerSecond', samples_per_second)
        logger.record_tabular('CumCompletedTrajs', self._cum_completed_trajs)
        logger.record_tabular('NewCompletedTrajs', self._new_completed_trajs)
        logger.record_tabular('StepsInTrajWindow',
            sum(info["Length"] for info in self._traj_infos))
        self._log_infos()

        self._last_time = new_time
        logger.dump_tabular(with_prefix=False)

        self._new_completed_trajs = 0
        if itr < self.n_itr - 1:
            logger.log(f"Optimizing over {self.log_interval_itrs} iterations.")
            self.pbar = ProgBarCounter(self.log_interval_itrs)


class MinibatchRlEval(MinibatchRlBase):
    """Runs RL on minibatches; tracks performance offline using evaluation
    trajectories."""

    def train(self):
        n_itr = self.startup()
        with logger.prefix(f"itr #0 "):
            eval_traj_infos, eval_time = self.evaluate_agent(0)
            self.log_diagnostics(0, eval_traj_infos, eval_time)
        for itr in range(n_itr):
            with logger.prefix(f"itr #{itr} "):
                self.agent.sample_mode(itr)
                samples, traj_infos = self.sampler.obtain_samples(itr)
                self.agent.train_mode(itr)
                opt_info = self.algo.optimize_agent(itr, samples)
                self.store_diagnostics(itr, traj_infos, opt_info)
                if (itr + 1) % self.log_interval_itrs == 0:
                    eval_traj_infos, eval_time = self.evaluate_agent(itr)
                    self.log_diagnostics(itr, eval_traj_infos, eval_time)
        self.shutdown()

    def evaluate_agent(self, itr):
        if itr > 0:
            self.pbar.stop()
        logger.log("Evaluating agent...")
        self.agent.eval_mode(itr)  # Might be agent in sampler.
        eval_time = -time.time()
        traj_infos = self.sampler.evaluate_agent(itr)
        eval_time += time.time()
        logger.log("Evaluation run complete.")
        return traj_infos, eval_time

    def initialize_logging(self):
        self.cum_train_time = 0
        self.cum_eval_time = 0
        self.cum_total_time = 0
        super().initialize_logging()

    def store_diagnostics(self, itr, traj_infos, opt_info):
        for k, v in self._opt_infos.items():
            new_v = getattr(opt_info, k, [])
            v.extend(new_v if isinstance(new_v, list) else [new_v])
        self.pbar.update((itr + 1) % self.log_interval_itrs)

    def log_diagnostics(self, itr, eval_traj_infos, eval_time):
        self.save_itr_snapshot(itr)
        if not eval_traj_infos:
            logger.log("WARNING: had no complete trajectories in eval.")
        steps_in_eval = sum([info["Length"] for info in eval_traj_infos])
        logger.record_tabular('Iteration', itr)
        logger.record_tabular('CumSteps', itr * self.itr_batch_size)
        logger.record_tabular('StepsInEval', steps_in_eval)
        logger.record_tabular('TrajsInEval', len(eval_traj_infos))

        self._log_infos(eval_traj_infos)

        new_time = time.time()
        log_interval_time = new_time - self._last_time
        new_train_time = log_interval_time - eval_time
        self.cum_train_time += new_train_time
        self.cum_eval_time += eval_time
        self.cum_total_time += log_interval_time
        self._last_time = new_time
        samples_per_second = (float('nan') if itr == 0 else
            self.log_interval_itrs * self.itr_batch_size / new_train_time)

        logger.record_tabular('CumTrainTime', self.cum_train_time)
        logger.record_tabular('CumEvalTime', self.cum_eval_time)
        logger.record_tabular('CumTotalTime', self.cum_total_time)
        logger.record_tabular('SamplesPerSecond', samples_per_second)

        logger.dump_tabular(with_prefix=False)

        logger.log(f"optimizing over {self.log_interval_itrs} iterations")
        self.pbar = ProgBarCounter(self.log_interval_itrs)
