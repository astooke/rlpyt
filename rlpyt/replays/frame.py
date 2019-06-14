

from rlpyt.utils.buffer import buffer_from_example, get_leading_dims
from rlpyt.utils.collections import namedarraytuple


ReplaySamples = None


class FrameBufferMixin(object):
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

    def __init__(self, example, **kwargs):
        field_names = [f for f in example._fields if f != "observation"]
        global ReplaySamples
        ReplaySamples = namedarraytuple("ReplaySamples", field_names)
        replay_example = ReplaySamples(*(v for k, v in example.items()
            if k != "observation"))
        super().__init__(example=replay_example, **kwargs)
        # Equivalent to image.shape[0] if observation is image array (C,H,W):
        self.n_frames = n_frames = get_leading_dims(example.observation,
            n_dim=1)[0]
        # frames: oldest stored at t; newest_frames: shifted so newest stored at t.
        self.samples_frames = buffer_from_example(example.observation[0],
            (self.T + n_frames - 1, self.B))  # n-minus-1 frames duplicated.
        self.samples_new_frames = self.samples_frames[n_frames:]
        self.off_forward = max(self.off_forward, max(1, self.n_frames - 1))

    def append_samples(self, samples):
        t, fm1 = self.t, self.n_frames - 1
        replay_samples = ReplaySamples(*(v for k, v in samples.items()
            if k != "observation"))
        T, idxs = super().append_samples(replay_samples)
        self.samples_new_frames[idxs] = samples.observation[:, :, -1]
        if t == 0:  # Starting: write early frames
            for f in range(fm1):
                self.samples_frames[f] = samples.observation[0, :, f]
        elif self.t < t:  # Wrapped: write duplicate frames.
            self.samples_frames[:fm1] = self.samples_frames[-fm1:]
        return T, idxs
