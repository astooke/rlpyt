
import numpy as np

from rlpyt.replays.non_sequence.n_step import NStepReturnBuffer
from rlpyt.replays.frame import BaseFrameBuffer


class NStepFrameBuffer(BaseFrameBuffer, NStepReturnBuffer):

    def extract_observation(self, T_idxs, B_idxs):
        """Frames are returned OLDEST to NEWEST."""
        return np.stack([self.samples_frames[t:t + self.n_frames, b]
            for t, b in zip(T_idxs, B_idxs)], axis=0)  # [B,C,H,W]
