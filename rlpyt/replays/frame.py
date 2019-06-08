

from rlpyt.replays.n_step import BaseNStepReturnBuffer
from rlpyt.utils.buffer import buffer_from_example, get_leading_dims
from rlpyt.utils.collections import namedarraytuple


ReplaySamples = None


class BaseFrameBuffer(BaseNStepReturnBuffer):
    """
    Like n-step return buffer but expects multi-frame input observation where
    each new observation has one new frame and the rest old; stores only
    unique frames to save memory.  Samples observation should be shaped:
    [T,B,C,..] with C the number of frames.  Expects frame order: OLDEST to
    NEWEST.

    Latest n_steps up to cursor invalid as "now" because "next" not
    yet written.  Cursor invalid as "now" because previous action and
    reward overwritten.  NEW: Next n_frames-1 invalid as "now" because
    observation frames overwritten.
    """

    def __init__(self, example, *args, **kwargs):
        field_names = [f for f in example._fields if f != "observation"]
        global ReplaySamples
        ReplaySamples = namedarraytuple("ReplaySamples", field_names)
        replay_example = ReplaySamples(*(v for k, v in example.items()
            if k != "observation"))
        super().__init__(replay_example, *args, **kwargs)
        # Equivalent to image.shape[0] if observation is image array (C,H,W):
        self.n_frames = n_frames = get_leading_dims(example.observation, n_dim=1)
        self.samples_frames = buffer_from_example(example.observation[0],
            (self.T + n_frames - 1, self.B))
        self.off_forward = max(1, self.n_frames - 1)

    def append_samples(self, samples):
        T, B = get_leading_dims(samples, n_dim=2)
        t, fm1 = self.t, self.n_frames - 1
        num_t = min(T, self.T - t)
        # Oldest frame at time t stored at frames[t], so observation is frames[t:t+fm1].
        # Wrap frames duplicated: frames[:fm1] == frames[-fm1:]; len(frames) == T + fm1.
        # Write new frames (-1 idx for newest frame at time t):
        self.samples_frames[t + fm1:t + fm1 + num_t] = samples.observation[:num_t, :, -1]
        if num_t < T:  # Edge case: wrap end of buffer.
            self.samples_frames[:T - num_t + fm1] = samples.observation[num_t - fm1:, :, -1]
        elif t == 0:  # Edge case: write earlier frames.
            self.samples_frames[:fm1] = samples.observation[0, :, -fm1:]
        replay_samples = ReplaySamples(*(v for k, v in samples.items()
            if k != "observation"))
        return super().append_samples(replay_samples)
