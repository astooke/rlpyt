import numpy as np

from rlpyt.replays.sequence.n_step import SequenceNStepReturnBuffer
from rlpyt.replays.frame import FrameBufferMixin
from rlpyt.replays.sequence.uniform import UniformSequenceReplay
from rlpyt.replays.sequence.prioritized import PrioritizedSequenceReplay


class SequenceNStepFrameBuffer(FrameBufferMixin, SequenceNStepReturnBuffer):

    def extract_observation(self, T_idxs, B_idxs, T):
        """Frames are returned OLDEST to NEWEST."""
        observation = np.empty(shape=(T, len(B_idxs), self.n_frames) +  # [T,B,C,H,W]
            self.samples_frames.shape[2:], dtype=self.samples_frames.dtype)
        for i, (t, b) in enumerate(zip(T_idxs, B_idxs)):
            if t + T >= self.T:  # wrap (n_frames duplicated)
                m = self.T - t
                w = T - m
                for f in range(self.n_frames):
                    observation[:m, i, f] = self.samples_frames[t + f:t + f + m, b]
                    observation[m:, i, f] = self.samples_frames[f:w + f, b]
            else:
                for f in range(self.n_frames):
                    observation[:, i, f] = self.samples_frames[t + f:t + f + T, b]
        return observation


class UniformSequenceReplayFrameBuffer(UniformSequenceReplay,
        SequenceNStepFrameBuffer):

    pass


class PrioritizedSequenceReplayFrameBuffer(PrioritizedSequenceReplay,
        SequenceNStepFrameBuffer):

    pass
