
from rlpyt.agents.base import BaseAgent
from rlpyt.utils.collections import namedarraytuple

AgentTrainInputs = namedarraytuple("AgentTrainInputs",
    ["observation", "prev_action", "prev_reward", "init_rnn_state"])


class BaseRecurrentAgent(BaseAgent):
    """Manages recurrent state during sampling, so sampler remains agnostic."""

    recurrent = True
    _prev_rnn_state = None

    def reset(self):
        self._prev_rnn_state = None  # Gets passed as None; module makes zeros.

    def reset_one(self, idx):
        self._reset_one(idx, self._prev_rnn_state)

    def _reset_one(self, idx, prev_rnn_state):
        """Assume each state is of shape: [B, ...], but can be nested tuples.
        Reset chosen index in the Batch dimension."""
        if isinstance(prev_rnn_state, tuple):
            for prev_state in prev_rnn_state:
                self._reset_one(idx, prev_state)
        elif prev_rnn_state is not None:
            prev_rnn_state[idx] = 0

    def advance_rnn_state(self, new_rnn_state):
        self._prev_rnn_state = new_rnn_state

    @property
    def prev_rnn_state(self):
        return self._prev_rnn_state
