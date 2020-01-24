
import numpy as np

from rlpyt.replays.non_sequence.n_step import NStepReturnBuffer
from rlpyt.replays.frame import FrameBufferMixin
from rlpyt.replays.non_sequence.uniform import UniformReplay
from rlpyt.replays.non_sequence.prioritized import PrioritizedReplay
from rlpyt.replays.async_ import AsyncReplayBufferMixin


class NStepFrameBuffer(FrameBufferMixin, NStepReturnBuffer):
    """Special method for re-assembling observations from frames."""

    def extract_observation(self, T_idxs, B_idxs):
        """Assembles multi-frame observations from frame-wise buffer.  Frames
        are ordered OLDEST to NEWEST along C dim: [B,C,H,W].  Where
        ``done=True`` is found, the history is not full due to recent
        environment reset, so these frames are zero-ed.
        """
        # Begin/end frames duplicated in samples_frames so no wrapping here.
        # return np.stack([self.samples_frames[t:t + self.n_frames, b]
        #     for t, b in zip(T_idxs, B_idxs)], axis=0)  # [B,C,H,W]
        observation = np.stack([self.samples_frames[t:t + self.n_frames, b]
            for t, b in zip(T_idxs, B_idxs)], axis=0)  # [B,C,H,W]
        # Populate empty (zero) frames after environment done.
        for f in range(1, self.n_frames):
            # e.g. if done 1 step prior, all but newest frame go blank.
            b_blanks = np.where(self.samples.done[T_idxs - f, B_idxs])[0]
            observation[b_blanks, :self.n_frames - f] = 0
        return observation


class UniformReplayFrameBuffer(UniformReplay, NStepFrameBuffer):
    pass


class PrioritizedReplayFrameBuffer(PrioritizedReplay, NStepFrameBuffer):
    pass


class AsyncUniformReplayFrameBuffer(AsyncReplayBufferMixin,
        UniformReplayFrameBuffer):
    pass


class AsyncPrioritizedReplayFrameBuffer(AsyncReplayBufferMixin,
        PrioritizedReplayFrameBuffer):
    pass
