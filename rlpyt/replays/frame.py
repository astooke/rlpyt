

from rlpyt.utils.buffer import buffer_from_example, get_leading_dims
from rlpyt.utils.collections import namedarraytuple
from rlpyt.utils.logging import logger

BufferSamples = None


class FrameBufferMixin:
    """
    Like n-step return buffer but expects multi-frame input observation where
    each new observation has one new frame and the rest old; stores only
    unique frames to save memory.  Samples observation should be shaped:
    [T,B,C,..] with C the number of frames.  Expects frame order: OLDEST to
    NEWEST.

    A special method for replay will be required to piece the frames back
    together into full observations.

    Latest n_steps up to cursor temporarilty invalid because "next" not yet
    written.  Cursor timestep invalid because previous action and reward
    overwritten.  NEW: Next n_frames-1 invalid because observation history
    frames overwritten.
    """

    def __init__(self, example, **kwargs):
        field_names = [f for f in example._fields if f != "observation"]
        global BufferSamples
        BufferSamples = namedarraytuple("BufferSamples", field_names)
        buffer_example = BufferSamples(*(v for k, v in example.items()
            if k != "observation"))
        super().__init__(example=buffer_example, **kwargs)
        # Equivalent to image.shape[0] if observation is image array (C,H,W):
        self.n_frames = n_frames = get_leading_dims(example.observation,
            n_dim=1)[0]
        logger.log(f"Frame-based buffer using {n_frames}-frame sequences.")
        # frames: oldest stored at t; duplicate n_frames - 1 beginning & end.
        self.samples_frames = buffer_from_example(example.observation[0],
            (self.T + n_frames - 1, self.B),
            share_memory=self.async_)  # [T+n_frames-1,B,H,W]
        # new_frames: shifted so newest stored at t; no duplication.
        self.samples_new_frames = self.samples_frames[n_frames - 1:]  # [T,B,H,W]
        self.off_forward = max(self.off_forward, n_frames - 1)

    def append_samples(self, samples):
        """Appends all samples except for the `observation` as normal.
        Only the new frame in each observation is recorded."""
        t, fm1 = self.t, self.n_frames - 1
        buffer_samples = BufferSamples(*(v for k, v in samples.items()
            if k != "observation"))
        T, idxs = super().append_samples(buffer_samples)
        self.samples_new_frames[idxs] = samples.observation[:, :, -1]
        if t == 0:  # Starting: write early frames
            for f in range(fm1):
                self.samples_frames[f] = samples.observation[0, :, f]
        elif self.t < t:  # Wrapped: copy duplicate frames.
            self.samples_frames[:fm1] = self.samples_frames[-fm1:]
        return T, idxs
