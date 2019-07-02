import torch

from rlpyt.utils.quick_args import save__init__args
from rlpyt.utils.collections import namedarraytuple
from rlpyt.utils.synchronize import RWLock

AgentInputs = namedarraytuple("AgentInputs",
    ["observation", "prev_action", "prev_reward"])
AgentStep = namedarraytuple("AgentStep", ["action", "agent_info"])


class BaseAgent(object):

    model = None  # type: torch.nn.Module
    shared_model = None
    device = torch.device("cpu")
    recurrent = False
    _mode = None

    def __init__(self, ModelCls, model_kwargs, initial_model_state_dict=None):
        save__init__args(locals())
        self.env_model_kwargs = dict()  # Populate in initialize().
        self._rw_lock = RWLock()  # For async runner.

    def __call__(self, observation, prev_action, prev_reward):
        """Returns values from model forward pass on training data."""
        raise NotImplementedError

    def initialize(self, env_spaces, share_memory=False):
        """Builds the model."""
        raise NotImplementedError

    def initialize_cuda(self, cuda_idx=None):
        """Call after initialize() and after forking sampler workers, but
        before initializing algorithm."""
        raise NotImplementedError

    @torch.no_grad()  # Hint: apply this decorator on overriding method.
    def step(self, observation, prev_action, prev_reward):
        raise NotImplementedError  # return type: AgentStep

    def reset(self):
        pass

    def reset_one(self, idx):
        pass

    def parameters(self):
        """Parameters to be optimized (overwrite in subclass if multiple models)."""
        return self.model.parameters()

    def state_dict(self):
        """Parameters for saving."""
        return self.model.state_dict()

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)

    def train_mode(self, itr):
        """Go into training mode."""
        self.model.train()
        self._mode = "train"

    def sample_mode(self, itr):
        """Go into sampling mode."""
        self.model.eval()
        self._mode = "sample"

    def eval_mode(self, itr):
        """Go into evaluation mode."""
        self.model.eval()
        self._mode = "eval"

    def sync_shared_memory(self):
        """Call in sampler master, after share_memory=True to initialize()."""
        if self.shared_model is not self.model:  # (self.model gets trained)
            self.shared_model.load_state_dict(self.model.state_dict())

    def send_shared_memory(self):
        with self._rw_lock.write_lock:
            self.sync_shared_memory()

    def recv_shared_memory(self):
        if self.shared_model is not self.model:
            with self._rw_lock:
                self.model.load_state_dict(self.shared_model.state_dict())


AgentInputsRnn = namedarraytuple("AgentInputsRnn",  # Training only.
    ["observation", "prev_action", "prev_reward", "init_rnn_state"])


class RecurrentAgentMixin(object):
    """Manages recurrent state during sampling, so sampler remains agnostic."""

    recurrent = True
    _prev_rnn_state = None
    _sample_rnn_state = None  # Store during eval.

    def reset(self):
        self._prev_rnn_state = None  # Gets passed as None; module makes zeros.

    def reset_one(self, idx):
        # Assume rnn_state is cudnn-compatible shape: [N,B,H]
        if self._prev_rnn_state is not None:
            self._prev_rnn_state[:, idx] = 0  # Automatic recursion in namedarraytuple.

    def advance_rnn_state(self, new_rnn_state):
        self._prev_rnn_state = new_rnn_state

    @property
    def prev_rnn_state(self):
        return self._prev_rnn_state

    def train_mode(self, itr):
        if self._mode == "sample":
            self._sample_rnn_state = self._prev_rnn_state
        super().train_mode(itr)

    def sample_mode(self, itr):
        if self._mode != "sample":
            self._prev_rnn_state = self._sample_rnn_state
        super().sample_mode(itr)

    def eval_mode(self, itr):
        if self._mode == "sample":
            self._sample_rnn_state = self._prev_rnn_state
        self._prev_rnn_state = None
        super().eval_mode(itr)
