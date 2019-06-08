
from rlpyt.replays.n_step import BaseNStepReturnBuffer
from rlpyt.agents.base import AgentInputs
from rlpyt.utils.collections import namedarraytuple
from rlpyt.utils.buffer import torchify_buffer

SamplesFromReplay = namedarraytuple("SamplesFromReplay",
    ["agent_inputs", "action", "return_", "done_n", "next_agent_inputs"])


class NStepReturnBuffer(BaseNStepReturnBuffer):

    def extract_batch(self, T_idxs, B_idxs):
        s = self.samples
        next_T_idxs = (T_idxs + self.n_step_return) % self.T
        batch = SamplesFromReplay(
            agent_inputs=AgentInputs(
                observation=self.extract_observation(T_idxs, B_idxs),
                prev_action=s.action[T_idxs - 1, B_idxs],
                prev_reward=s.reward[T_idxs - 1, B_idxs],
            ),
            action=s.action[T_idxs, B_idxs],
            return_=self.samples_return_[T_idxs, B_idxs],
            done_n=self.samples_done_n[T_idxs, B_idxs],
            next_agent_inputs=AgentInputs(
                observation=self.extract_observation(next_T_idxs, B_idxs),
                prev_action=s.action[next_T_idxs - 1, B_idxs],
                prev_reward=s.reward[next_T_idxs - 1, B_idxs],
            ),
        )
        return torchify_buffer(batch)

    def extract_observation(self, T_idxs, B_idxs):
        """Generalization anticipating frame-based buffer."""
        return self.samples.observation[T_idxs, B_idxs]
