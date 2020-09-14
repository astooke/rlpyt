
import pickle
import os.path as osp
import time

from rlpyt.runners.minibatch_rl import MinibatchRl, MinibatchRlEval
from rlpyt.utils.logging import logger
from rlpyt.utils.prog_bar import ProgBarCounter

from rlpyt.ul.runners.envstep_runner import MinibatchRlEvalEnvStep

class ReplaySaverMixin:

    def log_diagnostics(self, itr, *args, **kwargs):
        if itr > 0:
            logger.log("Saving replay buffer...")
            cum_steps = (itr + 1) * self.sampler.batch_size * self.world_size
            snapshot_mode = logger.get_snapshot_mode()
            if snapshot_mode == "all":
                filename = f"replaybuffer_{cum_steps}.pkl"
            elif snapshot_mode == "last":
                filename = "replaybuffer.pkl"
            else:
                raise NotImplementedError
            filename = osp.join(logger.get_snapshot_dir(), filename)
            with open(filename, "wb") as fh:
                pickle.dump(self.algo.replay_buffer, fh, protocol=4)
            logger.log("Replay buffer saved.")
        super().log_diagnostics(itr, *args, **kwargs)


class MinibatchRlReplaySaver(ReplaySaverMixin, MinibatchRl):
    pass


class MinibatchRlEvalReplaySaver(ReplaySaverMixin, MinibatchRlEval):
    pass


class MinibatchRlEvalEnvStepReplaySaver(ReplaySaverMixin, MinibatchRlEvalEnvStep):
    pass
