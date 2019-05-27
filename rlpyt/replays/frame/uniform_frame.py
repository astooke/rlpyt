
from rlpyt.replays.frame.n_step_frame import NStepFrameBuffer
from rlpyt.replays.uniform import UniformReplayBuffer


class UniformFrameReplayBuffer(NStepFrameBuffer, UniformReplayBuffer):

    def sample_batch(self, batch_size):
        return UniformReplayBuffer.sample_batch(self, batch_size)
