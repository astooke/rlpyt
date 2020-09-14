
import time

from rlpyt.runners.minibatch_rl import MinibatchRlEval
from rlpyt.utils.logging import logger
from rlpyt.utils.prog_bar import ProgBarCounter


class MinibatchRlEvalEnvStep(MinibatchRlEval):

    def __init__(self, *args, frame_skip=1, **kwargs):
        super().__init__(*args, **kwargs)
        self._frame_skip = frame_skip

    def log_diagnostics(self, itr, eval_traj_infos, eval_time):
        """
        Write diagnostics (including stored ones) to csv via the logger.
        ONE NEW LINE VS REGULAR RUNNER, TO LOG ENV STEPS = steps*frame_skip
        """
        if not eval_traj_infos:
            logger.log("WARNING: had no complete trajectories in eval.")
        steps_in_eval = sum([info["Length"] for info in eval_traj_infos])
        logger.record_tabular('StepsInEval', steps_in_eval)
        logger.record_tabular('TrajsInEval', len(eval_traj_infos))
        self._cum_eval_time += eval_time
        logger.record_tabular('CumEvalTime', self._cum_eval_time)

        if itr > 0:
            self.pbar.stop()
        if itr >= self.min_itr_learn - 1:
            self.save_itr_snapshot(itr)
        new_time = time.time()
        self._cum_time = new_time - self._start_time
        train_time_elapsed = new_time - self._last_time - eval_time
        new_updates = self.algo.update_counter - self._last_update_counter
        new_samples = (self.sampler.batch_size * self.world_size *
            self.log_interval_itrs)
        updates_per_second = (float('nan') if itr == 0 else
            new_updates / train_time_elapsed)
        samples_per_second = (float('nan') if itr == 0 else
            new_samples / train_time_elapsed)
        replay_ratio = (new_updates * self.algo.batch_size * self.world_size /
            new_samples)
        cum_replay_ratio = (self.algo.batch_size * self.algo.update_counter /
            ((itr + 1) * self.sampler.batch_size))  # world_size cancels.
        cum_steps = (itr + 1) * self.sampler.batch_size * self.world_size

        if self._eval:
            logger.record_tabular('CumTrainTime',
                self._cum_time - self._cum_eval_time)  # Already added new eval_time.
        logger.record_tabular('Iteration', itr)
        logger.record_tabular('CumTime (s)', self._cum_time)
        logger.record_tabular('CumSteps', cum_steps)
        logger.record_tabular('EnvSteps', cum_steps * self._frame_skip)  # NEW LINE
        logger.record_tabular('CumCompletedTrajs', self._cum_completed_trajs)
        logger.record_tabular('CumUpdates', self.algo.update_counter)
        logger.record_tabular('StepsPerSecond', samples_per_second)
        logger.record_tabular('UpdatesPerSecond', updates_per_second)
        logger.record_tabular('ReplayRatio', replay_ratio)
        logger.record_tabular('CumReplayRatio', cum_replay_ratio)
        self._log_infos(eval_traj_infos)
        logger.dump_tabular(with_prefix=False)

        self._last_time = new_time
        self._last_update_counter = self.algo.update_counter
        if itr < self.n_itr - 1:
            logger.log(f"Optimizing over {self.log_interval_itrs} iterations.")
            self.pbar = ProgBarCounter(self.log_interval_itrs)
