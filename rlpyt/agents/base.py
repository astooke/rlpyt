
import torch

from rlpyt.utils.quick_args import save__init__args
from rlpyt.utils.collections import namedarraytuple

AgentInput = namedarraytuple("AgentInput",
    ["observation", "prev_action", "prev_reward"])
AgentStep = namedarraytuple("AgentStep", ["action", "agent_info"])
AgentInfo = namedarraytuple("AgentInfo", [])


class BaseAgent(object):

    model = None  # type: torch.nn.Module
    shared_model = None
    device = torch.device("cpu")
    recurrent = False

    def __init__(self, ModelCls, model_kwargs, initial_model_state_dict=None):
        save__init__args(locals())
        self.env_model_kwargs = dict()  # Populate in initialize().

    def __call__(self, train_samples):
        """Returns values from model forward pass on training data."""
        raise NotImplementedError

    def initialize(self, env_spec, share_memory=False):
        """Builds the model."""
        raise NotImplementedError

    def initialize_cuda(self, cuda_idx=None):
        """Call after initialize() and after forking sampler workers."""
        if cuda_idx is None:
            return   # CPU
        if self._memory_is_shared:
            self.shared_model = self.model
            self.model = self.ModelCls(**self.env_model_kwargs, **self.model_kwargs)
            self.model.load_state_dict(self.shared_model.state_dict())
        self.device = torch.device("cuda", index=cuda_idx)
        self.model.to(self.device)

    @torch.no_grad()  # Hint: apply this decorator on overriding method.
    def step(self, observation, prev_action, prev_reward):
        raise NotImplementedError  # return types: action, AgentInfo

    def reset(self):
        pass

    def reset_one(self, idx):
        pass

    @property
    def recurrent(self):
        return False

    def sync_shared_memory(self):
        """Call in sampler master, after share_memory=True to initialize()."""
        if self.shared_model is not None:
            self.shared_model.load_state_dict(self.model.state_dict())


RecurrentAgentInfo = namedarraytuple("AgentInfo", ["prev_rnn_state"])


class BaseRecurrentAgent(BaseAgent):

    _prev_rnn_state = None
    recurrent = True

    def reset(self):
        self._prev_rnn_state = None  # Gets passed as None; module makes zeros.

    def reset_one(self, idx):
        self._reset_one(idx, self._prev_rnn_state)

    def _reset_one(self, idx, prev_rnn_state):
        """Assume each state is of shape: [N, B, H], but can be nested
        list/tuple. Reset chosen index in the B dimension."""
        if isinstance(prev_rnn_state, (list, tuple)):
            for prev_state in prev_rnn_state:
                self._reset_one(prev_rnn_state)
        elif prev_rnn_state is not None:
            prev_rnn_state[:, idx] = 0.

    def advance_rnn_state(self, new_rnn_state):
        self._prev_rnn_state = new_rnn_state

    @property
    def prev_rnn_state(self):
        return self._prev_rnn_state
