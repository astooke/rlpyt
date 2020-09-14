"""
Methods to overwrite for the saved replay buffer, to return
different samples than was used by the replay buffer object
used to collect the samples.
"""
import numpy as np

from rlpyt.utils.buffer import torchify_buffer, buffer_func
from rlpyt.utils.misc import extract_sequences
from rlpyt.utils.collections import namedarraytuple
from rlpyt.utils.logging import logger

SamplesFromReplay = namedarraytuple("SamplesFromReplay",
    ["observation", "action", "reward", "done", "prev_action", "prev_reward"])
SamplesFromReplayPC = namedarraytuple("SamplesFromReplayPC",
    SamplesFromReplay._fields + ("pixctl_return",))


class UlForRlReplayBuffer:

    def __init__(
            self,
            replay_buffer,
            replay_T=1,
            validation_split=0.0,
            pixel_control_buffer=None,
        ):
        self.load_replay(replay_buffer, pixel_control_buffer)
        self.replay_T = replay_T
        self.validation_t = int((self.T - replay_T) * (1 - validation_split))
        if pixel_control_buffer is not None:
            logger.log("Replay buffer receiving pixel control returns.")

    def load_replay(self, replay_buffer, pixel_control_buffer=None):
        if isinstance(replay_buffer, (tuple, list)):
            self._load_multiple_replays(replay_buffer, pixel_control_buffer)
        else:
            self._load_single_replay(replay_buffer, pixel_control_buffer)

    def _load_single_replay(self, replay_buffer, pixel_control_buffer=None):
        # Make sure the unpickled replay buffer is exactly full.
        assert replay_buffer.t == 0  # no wrapping
        assert replay_buffer._buffer_full
        self.loaded_buffer = replay_buffer
        self._samples = self.loaded_buffer.samples
        self._is_frame_buffer = hasattr(replay_buffer, "samples_frames")
        if self._is_frame_buffer:
            self.n_frames = self.loaded_buffer.n_frames
            self._samples_frames = self.loaded_buffer.samples_frames

        self.T, self.B = self.samples.reward.shape
        self.size = self.T * self.B
        self.pixel_control_buffer = pixel_control_buffer

    def _load_multiple_replays(self, replay_buffers, pixel_control_buffers=None):
        """Now replay_buffer is actually a tuple of them."""
        assert isinstance(replay_buffers, (tuple, list))
        if pixel_control_buffers is not None:
            assert isinstance(pixel_control_buffers, (tuple, list))
            assert len(replay_buffers) == len(pixel_control_buffers)

        # Make sure the unpickled replay buffers are exactly full.
        T = replay_buffers[0].T
        self._is_frame_buffer = hasattr(replay_buffers[0], "samples_frames")
        if self._is_frame_buffer:
            self.n_frames = replay_buffers[0].n_frames
        for rep in replay_buffers:
            assert rep.t == 0  # no wrapping
            assert rep._buffer_full  # try to make sure it's real data
            assert rep.T == T  # make sure they are the same time length
            assert hasattr(rep, "samples_frames") == self._is_frame_buffer
            if self._is_frame_buffer:
                assert rep.n_frames == self.n_frames
            # (can be different B, though)
        # Load from each replay for each field, concatenating along B dimension.
        self._samples = buffer_concatenate(  # main samples
            tuple(rep.samples for rep in replay_buffers), axis=1)
        if self._is_frame_buffer:
            self._samples_frames = buffer_concatenate(
                tuple(rep.samples_frames for rep in replay_buffers), axis=1)
        del replay_buffers  # Otherwise would hold double memory
        self.T, self.B = self.samples.reward.shape
        # self.B = sum(rep.B for rep in replay_buffers)
        self.size = self.T * self.B

        self.pixel_control_buffer = None
        if pixel_control_buffers is not None:
            pixctl_reward = buffer_concatenate(tuple(pcb["reward"]
                for pcb in pixel_control_buffers), axis=1)
            pixctl_return = buffer_concatenate(tuple(pcb["return_"]
                for pcb in pixel_control_buffers), axis=1)
            self.pixel_control_buffer = dict(
                reward=pixctl_reward,
                return_=pixctl_return,
            )
            del pixel_control_buffers

    @property
    def samples(self):
        return self._samples

    @property
    def samples_frames(self):
        return self._samples_frames

    def get_examples(self):
        """To use when initializing NN model."""
        observation = (self.samples_frames[:self.n_frames, 0]
            if self._is_frame_buffer else
            self.samples.observation[0, 0])
        examples = SamplesFromReplay(
            observation=observation,
            action=self.samples.action[0, 0],
            reward=self.samples.reward[0, 0],
            done=self.samples.done[0, 0],
            prev_action=self.samples.action[0, 0],
            prev_reward=self.samples.reward[0, 0],
        )
        if self.pixel_control_buffer is not None:
            examples = SamplesFromReplayPC(*examples,
                pixctl_return=self.pixel_control_buffer["return_"][0, 0])
        return examples

    def sample_batch(self, batch_B, validation=False):
        T_idxs, B_idxs = self.sample_idxs(batch_B, validation=False)
        return self.extract_batch(T_idxs, B_idxs)

    def sample_idxs(self, batch_B, validation=False):
        """Uniform replay."""
        if validation:
            low = self.validation_t
            high = self.T - self.replay_T
        else:
            low = self.n_frames - 1 if self._is_frame_buffer else 0
            high = self.validation_t - self.replay_T
        high = self.T - self.replay_T
        T_idxs = np.random.randint(low=low, high=high, size=(batch_B,))
        B_idxs = np.random.randint(low=0, high=self.B, size=(batch_B,))
        return T_idxs, B_idxs

    def extract_batch(self, T_idxs, B_idxs):
        T = self.replay_T
        all_action = buffer_func(self.samples.action, extract_sequences,
            T_idxs - 1, B_idxs, T + 1)
        all_reward = extract_sequences(self.samples.reward,
            T_idxs - 1, B_idxs, T + 1)
        batch = SamplesFromReplay(
            observation=self.extract_observation(T_idxs, B_idxs),
            action=all_action[1:],
            reward=all_reward[1:],
            done=extract_sequences(self.samples.done, T_idxs, B_idxs, T),
            prev_action=all_action[:-1],
            prev_reward=all_reward[:-1],
        )
        if self.pixel_control_buffer is not None:
            pixctl_return = extract_sequences(
                self.pixel_control_buffer["return_"],
                T_idxs, B_idxs, T)
            batch = SamplesFromReplayPC(*batch,
                pixctl_return=pixctl_return)
        return torchify_buffer(batch)

    def extract_observation(self, T_idxs, B_idxs):
        T = self.replay_T
        if not self._is_frame_buffer:
            return buffer_func(self.samples.observation,
                extract_sequences, T_idxs, B_idxs, T)
        frames = self.samples_frames
        observation = np.empty(
            shape=(T, len(B_idxs), self.n_frames) + frames.shape[2:],  # [T,B,C,H,W]
            dtype=frames.dtype,
        )
        fm1 = self.n_frames - 1
        for i, (t, b) in enumerate(zip(T_idxs, B_idxs)):
            assert t + T <= self.T  # no wrapping allowed
            for f in range(self.n_frames):
                observation[:, i, f] = frames[t + f:t + f + T, b]

            # Populate empty (zero) frames after environment done.
            assert t - fm1 >= 0  # no wrapping allowed
            done_idxs = slice(t - fm1, t + T)
            done_fm1 = self.samples.done[done_idxs, b]
            if np.any(done_fm1):
                where_done_t = np.where(done_fm1)[0] - fm1  # Might be negative...
                for f in range(1, self.n_frames):
                    t_blanks = where_done_t + f  # ...might be > T...
                    t_blanks = t_blanks[(t_blanks >= 0) & (t_blanks < T)]  # ..don't let it wrap.
                    observation[t_blanks, i, :self.n_frames - f] = 0
        return observation


def buffer_concatenate(buffers, axis=0):
    assert type(buffers) == tuple
    if isinstance(buffers[0], np.ndarray):
        try:
            return np.concatenate(buffers, axis=axis)
        except ValueError:
            logger.log("Had a ValueError in buffer concat, probably action dimensions that don't line up, populating with zeros.")
            logger.log(f"buffer shapes: {[buf.shape for buf in buffers]}")
            return np.zeros((buffers[0].shape[0], sum(buf.shape[1] for buf in buffers)))
    fields = buffers[0]._fields
    for buf in buffers:
        # try to make sure they're the same structure
        assert buf._fields == fields
    new_buf = buffers[0]
    fields = new_buf._fields
    new_buf = new_buf._make(tuple(
        buffer_concatenate(tuple(getattr(buf, field) for buf in buffers), axis=1)
            for field in fields))
    return new_buf
