
import torch

from rlpyt.utils.quick_args import save_args
from rlpyt.uitls.collections import namedarraytuple


AgentStep = namedarraytuple("AgentStep",
    ["action", "dist_info", "value"])
AgentTrain = namedarraytuple("AgentTrain",
    ["dist_info", "value"])


class BaseAgent(object):

    model = None  # type: torch.nn.Module
    _shared_model = None
    distribution = None  # type: Distribution

    def __init__(self, ModelCls, model_kwargs, initial_state_dict=None):
        save_args(locals(), underscore=True)

    def __call__(self, samples):
        raise NotImplementedError  # return type: AgentTrain

    def initialize(self, env_spec, share_memory=False):
        self._env_spec = env_spec
        self.model = self._ModelCls(env_spec, **self._model_kwargs)
        if share_memory:
            self.model.share_memory()
            self._shared_memory = share_memory
        if self._initial_state_dict is not None:
            self.model.load_state_dict(self._initial_state_dict)

    def intialize_cuda(self, cuda_idx=None):
        """Call after initialize and after forking sampler workers."""
        self._cuda_idx = cuda_idx
        if cuda_idx is None:
            return   # CPU
        if self._shared_memory:
            self._shared_model = self.model
            self.model = self._ModelCls(self._env_spec, **self._model_kwargs)
            self.model.load_state_dict(self._shared_model.state_dict())
        self.model.to("cuda:" + str(cuda_idx))

    @torch.no_grad()  # Hint: apply this decorator on overriding method.
    def sample_action(self, observation, prev_action, prev_reward):
        raise NotImplementedError  # return types: AgentStep

    def reset(self):
        pass

    def reset_one(self, idx):
        pass

    @property
    def recurrent(self):
        return False

    def sync_shared_memory(self):
        """Call in sampler master, after share_memory=True to initialize()."""
        if self._shared_model is not None:
            self._shared_model.load_state_dict(self.model.state_dict())


RecurrentAgentStep = namedarraytuple("AgentStep",
    ["action", "dist_info", "value", "prev_rnn_state"])


class BaseRecurrentAgent(BaseAgent):

    _prev_rnn_state = None

    @property
    def recurrent(self):
        return True

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
