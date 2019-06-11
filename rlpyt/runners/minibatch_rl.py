
import time
from collections import deque

from rlpyt.runners.minibatch_rl_base import MinibatchRlBase
from rlpyt.utils.logging import logger
from rlpyt.utils.prog_bar import ProgBarCounter


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
                self.agent.sample_mode()  # Might not be this agent sampling.
                samples, traj_infos = self.sampler.obtain_samples(itr)
                self.agent.train_mode()
                _opt_data, opt_info = self.algo.optimize_agent(samples, itr)
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
            self.sampler.batch_spec.size) / (new_time - self._last_time)
        logger.record_tabular('Iteration', itr)
        logger.record_tabular('CumTotalSteps', (itr + 1) * self.sampler.batch_spec.size)
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
