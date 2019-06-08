
import numpy as np

from rlpyt.replays.n_step import BaseNStepReturnBuffer
from rlpyt.agents.base_recurrent import AgentTrainInputs
from rlpyt.utils.buffer import torchify_buffer
from rlpyt.utils.collections import namedarraytuple

SamplesFromReplay = namedarraytuple("SamplesFromReplay",
    ["agent_inputs", "action", "return_", "done", "done_n", "next_agent_inputs"])


class SequenceNStepReturnBuffer(BaseNStepReturnBuffer):

    def extract_batch(self, T_idxs, B_idxs, T):
        s = self.samples
        n_T_idxs = (T_idxs + self.n_step_return) % self.T  # Can wrap.
        batch = SamplesFromReplay(
            agent_inputs=AgentTrainInputs(
                observation=self.extract_observation(T_idxs, B_idxs, T),
                prev_action=extract_sequences(s.action, T_idxs - 1, B_idxs, T),
                prev_reward=extract_sequences(s.reward, T_idxs - 1, B_idxs, T),
            ),
            init_rnn_state=s.prev_rnn_state[T_idxs, B_idxs],
            action=extract_sequences(s.action, T_idxs, B_idxs, T),
            return_=extract_sequences(self.samples_return_, T_idxs, B_idxs, T),
            done=extract_sequences(s.done, T_idxs, B_idxs, T),
            done_n=extract_sequences(self.samples_done_n, T_idxs, B_idxs, T),
            next_agent_inputs=AgentTrainInputs(
                observation=self.extract_observation(n_T_idxs, B_idxs, T),
                prev_action=extract_sequences(s.action, n_T_idxs - 1, B_idxs, T),
                prev_reward=extract_sequences(s.reward, n_T_idxs - 1, B_idxs, T),
            ),
            next_init_rnn_state=s.prev_rnn_state[n_T_idxs, B_idxs],
        )
        return torchify_buffer(batch)

    def extract_observation(self, T_idxs, B_idxs, T):
        """Generalization anticipating frame-buffer."""
        return extract_sequences(self.samples.observation, T_idxs, B_idxs, T)


def extract_sequences(array, T_idxs, B_idxs, T):
    sequences = np.empty(shape=(T, len(B_idxs)) + array.shape[2:],
        dtype=array.dtype)  # [T,B,..]
    sequences = list()
    for i, (t, b) in enumerate(zip(T_idxs, B_idxs)):
        if t + T >= len(array):  # wrap
            m = len(array) - t
            w = T - m
            sequences[:m, i] = array[t:, b]  # [m,..]
            sequences[m:, i] = array[:w, b]  # [w,..]
        else:
            sequences[:, i] = array[t:t + T, b]  # [T,..]
    return sequences
