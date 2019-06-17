
import numpy as np

from rlpyt.replays.non_sequence.n_step import NStepReturnBuffer
from rlpyt.replays.frame import FrameBufferMixin
from rlpyt.replays.non_sequence.uniform import UniformReplay
from rlpyt.replays.non_sequence.prioritized import PrioritizedReplay


class NStepFrameBuffer(FrameBufferMixin, NStepReturnBuffer):

    def extract_observation(self, T_idxs, B_idxs):
        """Frames are returned OLDEST to NEWEST."""
        # Begin/end frames duplicated in samples_frames so no wrapping here.
        # return np.stack([self.samples_frames[t:t + self.n_frames, b]
        #     for t, b in zip(T_idxs, B_idxs)], axis=0)  # [B,C,H,W]
        observation = np.stack([self.samples_frames[t:t + self.n_frames, b]
            for t, b in zip(T_idxs, B_idxs)], axis=0)  # [B,C,H,W]
        # Populate empty (zero) frames after environment done.
        for f in range(1, self.n_frames):
            # e.g. if done 1 step prior, all but newest frame go blank.
            done_idxs = np.where(self.samples.done[T_idxs - f, B_idxs])[0]
            observation[done_idxs, f - 1:-1] = 0
        return observation


class UniformReplayFrameBuffer(UniformReplay, NStepFrameBuffer):

    pass


class PrioritizedReplayFrameBuffer(PrioritizedReplay, NStepFrameBuffer):

    pass
