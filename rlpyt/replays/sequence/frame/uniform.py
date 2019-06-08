
from rlpyt.replays.sequence.frame.n_step import SequenceNStepFrameBuffer
from rlpyt.replays.sequence.uniform import UniformSequenceReplayBuffer


class UniformSequenceFrameBuffer(SequenceNStepFrameBuffer,
        UniformSequenceReplayBuffer):

    def sample_batch(self, batch_spec):
        return UniformSequenceReplayBuffer.sample_batch(self, batch_spec)
