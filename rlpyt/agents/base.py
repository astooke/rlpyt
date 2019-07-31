import multiprocessing as mp
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.parallel import DistributedDataParallelCPU as DDPC

from rlpyt.utils.quick_args import save__init__args
from rlpyt.utils.collections import namedarraytuple
from rlpyt.utils.synchronize import RWLock
from rlpyt.utils.logging import logger

AgentInputs = namedarraytuple("AgentInputs",
    ["observation", "prev_action", "prev_reward"])
AgentStep = namedarraytuple("AgentStep", ["action", "agent_info"])


class BaseAgent(object):

    model = None  # type: torch.nn.Module
    shared_model = None
    distribution = None
    device = torch.device("cpu")
    recurrent = False
    _mode = None

    def __init__(self, ModelCls, model_kwargs=None, initial_model_state_dict=None):
        model_kwargs = dict() if model_kwargs is None else model_kwargs
        save__init__args(locals())
        # The rest for async operations:
        self._rw_lock = RWLock()
        self._send_count = mp.RawValue("l", 0)
        self._recv_count = 0

    def __call__(self, observation, prev_action, prev_reward):
        """Returns values from model forward pass on training data."""
        raise NotImplementedError

    def initialize(self, env_spaces, share_memory=False, **kwargs):
        self.env_model_kwargs = self.make_env_to_model_kwargs(env_spaces)
        self.model = self.ModelCls(**self.env_model_kwargs,
            **self.model_kwargs)
        if share_memory:
            self.model.share_memory()
            self.shared_model = self.model
        if self.initial_model_state_dict is not None:
            self.model.load_state_dict(self.initial_model_state_dict)
        self.env_spaces = env_spaces
        self.share_memory = share_memory

    def make_env_to_model_kwargs(self, env_spaces):
        return {}

    def to_device(self, cuda_idx=None):
        """Overwite/extend for format other than 'self.model' for network(s)."""
        if cuda_idx is None:
            return
        if self.shared_model is not None:
            self.model = self.ModelCls(**self.env_model_kwargs,
                **self.model_kwargs)
            self.model.load_state_dict(self.shared_model.state_dict())
        self.device = torch.device("cuda", index=cuda_idx)
        self.model.to(self.device)
        logger.log(f"Initialized agent model on device: {self.device}.")

    def data_parallel(self):
        """Overwrite/extend for format other than 'self.model' for network(s)
        which will have gradients through them."""
        if self.device.dtype == "cpu":
            self.model = DDPC(self.model)
            logger.log("Initialized DistributedDataParallelCPU agent model.")
        else:
            self.model = DDP(self.model)
            logger.log("Initialized DistributedDataParallel agent model on "
                f"device {self.device}.")

    def async_cpu(self, share_memory=True):
        """Shared model among sampler processes separate from shared model
        in optimizer process, so sampler can copy over under lock."""
        if self.device.type != "cpu":
            return
        assert self.shared_model is not None
        self.model = self.ModelCls(**self.env_model_kwargs,
            **self.model_kwargs)
        self.model.load_state_dict(self.shared_model.state_dict())
        if share_memory:  # Not needed in async_serial.
            self.model.share_memory()  # For CPU workers.
        logger.log("Initialized async CPU agent model.")

    def collector_initialize(self, global_B=1, env_ranks=None):
        """If need to initialize within CPU sampler (e.g. vector eps greedy)"""
        pass

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
        if self.shared_model is not self.model:
            with self._rw_lock.write_lock:
                # Workaround the fact that DistributedDataParallel prepends
                # 'module.' to every key, but the sampler models will not be
                # wrapped in DistributedDataParallel.
                # (Solution from PyTorch forums.)
                model_state_dict = self.model.state_dict()
                new_state_dict = type(model_state_dict)()
                for k, v in model_state_dict.items():
                    if k[:7] == "module.":
                        new_state_dict[k[7:]] = v
                    else:
                        new_state_dict[k] = v
                self.shared_model.load_state_dict(new_state_dict)
                self._send_count.value += 1

    def recv_shared_memory(self):
        if self.shared_model is not self.model:
            with self._rw_lock:
                if self._recv_count < self._send_count.value:
                    self.model.load_state_dict(self.shared_model.state_dict())
                    self._recv_count += 1


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
