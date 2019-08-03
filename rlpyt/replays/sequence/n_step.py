
import math
import numpy as np

from rlpyt.replays.n_step import BaseNStepReturnBuffer
from rlpyt.utils.buffer import torchify_buffer, buffer_from_example, buffer_func
from rlpyt.utils.misc import extract_sequences
from rlpyt.utils.collections import namedarraytuple

SamplesFromReplay = namedarraytuple("SamplesFromReplay",
    ["all_observation", "all_action", "all_reward", "return_", "done", "done_n",
    "init_rnn_state"])

SamplesToBuffer = None


class SequenceNStepReturnBuffer(BaseNStepReturnBuffer):

    def __init__(self, example, size, B, rnn_state_interval, batch_T=None, **kwargs):
        self.rnn_state_interval = rnn_state_interval
        self.batch_T = batch_T  # Maybe required fixed depending on replay type.
        if rnn_state_interval <= 1:  # Store no rnn state or every rnn state.
            buffer_example = example
        else:
            # Store some of rnn states; remove from samples.
            field_names = [f for f in example._fields if f != "prev_rnn_state"]
            global SamplesToBuffer
            SamplesToBuffer = namedarraytuple("SamplesToBuffer", field_names)
            buffer_example = SamplesToBuffer(*(v for k, v in example.items()
                if k != "prev_rnn_state"))
            size = B * rnn_state_interval * math.ceil(  # T as multiple of interval.
                math.ceil(size / B) / rnn_state_interval)
            self.samples_prev_rnn_state = buffer_from_example(example.prev_rnn_state,
                (size // (B * rnn_state_interval), B),
                share_memory=self.async_)
        super().__init__(example=buffer_example, size=size, B=B, **kwargs)
        if rnn_state_interval > 1:
            assert self.T % rnn_state_interval == 0
        self.rnn_T = self.T // rnn_state_interval

    def append_samples(self, samples):
        t, rsi = self.t, self.rnn_state_interval
        if rsi <= 1:  # All or no rnn states stored.
            return super().append_smaples(samples)
        buffer_samples = SamplesToBuffer(*(v for k, v in samples.items()
            if k != "prev_rnn_state"))
        T, idxs = super().append_samples(buffer_samples)
        start, stop = math.ceil(t / rsi), ((t + T - 1) // rsi) + 1
        offset = (rsi - t) % rsi
        if stop > self.rnn_T:  # Wrap.
            rnn_idxs = np.arange(start, stop) % self.rnn_T
        else:
            rnn_idxs = slice(start, stop)
        self.samples_prev_rnn_state[rnn_idxs] = samples.prev_rnn_state[offset::rsi]
        return T, idxs

    def extract_batch(self, T_idxs, B_idxs, T):
        """Return full sequence of each field which encompasses all subsequences
        to be used, so algorithm can make sub-sequences by slicing on device,
        for reduced memory usage."""
        s, rsi = self.samples, self.rnn_state_interval
        if rsi > 1:
            assert np.all(np.asarray(T_idxs) % rsi == 0)
            init_rnn_state = self.samples_prev_rnn_state[T_idxs // rsi, B_idxs]
        elif rsi == 1:
            init_rnn_state = self.samples.prev_rnn_state[T_idxs, B_idxs]
        else:  # rsi == 0
            init_rnn_state = None
        batch = SamplesFromReplay(
            all_observation=self.extract_observation(T_idxs, B_idxs,
                T + self.n_step_return),
            all_action=buffer_func(s.action, extract_sequences, T_idxs - 1, B_idxs,
                T + self.n_step_return),  # Starts at prev_action.
            all_reward=extract_sequences(s.reward, T_idxs - 1, B_idxs,
                T + self.n_step_return),  # Only prev_reward (agent + target).
            return_=extract_sequences(self.samples_return_, T_idxs, B_idxs, T),
            done=extract_sequences(s.done, T_idxs, B_idxs, T),
            done_n=extract_sequences(self.samples_done_n, T_idxs, B_idxs, T),
            init_rnn_state=init_rnn_state,  # (Same state for agent and target.)
        )
        # NOTE: Algo might need to make zero prev_action/prev_reward depending on done.
        return torchify_buffer(batch)

    def extract_observation(self, T_idxs, B_idxs, T):
        """Generalization anticipating frame-buffer."""
        return buffer_func(self.samples.observation, extract_sequences,
            T_idxs, B_idxs, T)
