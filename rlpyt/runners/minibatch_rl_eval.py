
import time

from rlpyt.runners.minibatch_rl_base import MinibatchRlBase
from rlpyt.utils.logging import logger
from rlpyt.utils.prog_bar import ProgBarCounter


class MinibatchRlEval(MinibatchRlBase):

    def train(self):
        n_itr = self.startup()
        for itr in range(n_itr):
            with logger.prefix(f"itr #{itr} "):
                if itr % self.log_interval_itrs == 0:
                    eval_traj_infos, eval_time = self.evaluate_agent(itr)
                    self.log_diagnostics(itr, eval_traj_infos, eval_time)
                self.agent.sample_mode()
                samples, traj_infos = self.sampler.obtain_samples(itr)
                self.agent.train_mode()
                _opt_data, opt_info = self.algo.optimize_agent(samples, itr)
                self.store_diagnostics(itr, traj_infos, opt_info)
        self.shutdown()

    def evaluate_agent(self, itr):
        self.pbar.stop()
        logger.log("Evaluating agent...")
        self.agent.eval_mode()  # Might be agent in sampler.
        eval_start_time = time.time()
        traj_infos = self.sampler.evaluate_agent(itr)
        eval_end_time = time.time()
        logger.log("Evaluation run complete.")
        return traj_infos, eval_end_time - eval_start_time

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
            logger.log("ERROR: had no complete trajectories in eval.")
        steps_in_eval = sum([info["Length"] for info in eval_traj_infos])
        logger.record_tabular('Iteration', itr)
        logger.record_tabular('CumCompletedSteps', itr * self.sampler.batch_spec.size)
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
        train_samples_per_second = (float('nan') if itr == 0 else
            self.log_interval_itrs * self.sampler.batch_spec.size / new_train_time)

        logger.record_tabular('CumTrainTime', self._cum_train_time)
        logger.record_tabular('CumEvalTime', self._cum_eval_time)
        logger.record_tabular('CumTotalTime', self._cum_total_time)
        logger.record_tabular('TrainSamplesPerSecond', train_samples_per_second)

        logger.dump_tabular(with_prefix=False)

        logger.log(f"optimizing over {self.log_interval_itrs} iterations")
        self.pbar = ProgBarCounter(self.log_interval_itrs)
