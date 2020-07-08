import multiprocessing as mp
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
# from torch.nn.parallel import DistributedDataParallelCPU as DDPC  # Deprecated

from rlpyt.utils.quick_args import save__init__args
from rlpyt.utils.collections import namedarraytuple
from rlpyt.utils.synchronize import RWLock
from rlpyt.utils.logging import logger
from rlpyt.models.utils import strip_ddp_state_dict

AgentInputs = namedarraytuple("AgentInputs",
    ["observation", "prev_action", "prev_reward"])
AgentStep = namedarraytuple("AgentStep", ["action", "agent_info"])


class BaseAgent:
    """
    The agent performs many functions, including: action-selection during
    sampling, returning policy-related values to use in training (e.g. action
    probabilities), storing recurrent state during sampling, managing model
    device, and performing model parameter communication between processes.
    The agent is both interfaces: sampler<-->neural network<-->algorithm.
    Typically, each algorithm and environment combination will require at
    least some of its own agent functionality.

    The base agent automatically carries out some of these roles.  It assumes
    there is one neural network model.  Agents using multiple models might
    need to extend certain funcionality to include those models, depending on
    how they are used.
    """
    
    recurrent = False
    alternating = False

    def __init__(self, ModelCls=None, model_kwargs=None, initial_model_state_dict=None):
        """
        Arguments are saved but no model initialization occurs.

        Args:
            ModelCls: The model class to be used.
            model_kwargs (optional): Any keyword arguments to pass when instantiating the model.
            initial_model_state_dict (optional): Initial model parameter values.
        """

        save__init__args(locals())
        self.model = None  # type: torch.nn.Module
        self.shared_model = None
        self.distribution = None
        self.device = torch.device("cpu")
        self._mode = None
        if self.model_kwargs is None:
            self.model_kwargs = dict()
        # The rest only for async operations:
        self._rw_lock = RWLock()
        self._send_count = mp.RawValue("l", 0)
        self._recv_count = 0

    def __call__(self, observation, prev_action, prev_reward):
        """Returns values from model forward pass on training data (i.e. used
        in algorithm)."""
        raise NotImplementedError

    def initialize(self, env_spaces, share_memory=False, **kwargs):
        """
        Instantiates the neural net model(s) according to the environment
        interfaces.  

        Uses shared memory as needed--e.g. in CpuSampler, workers have a copy
        of the agent for action-selection.  The workers automatically hold
        up-to-date parameters in ``model``, because they exist in shared
        memory, constructed here before worker processes fork. Agents with
        additional model components (beyond ``self.model``) for
        action-selection should extend this method to share those, as well.

        Typically called in the sampler during startup.

        Args:
            env_spaces: passed to ``make_env_to_model_kwargs()``, typically namedtuple of 'observation' and 'action'.
            share_memory (bool): whether to use shared memory for model parameters.
        """
        self.env_model_kwargs = self.make_env_to_model_kwargs(env_spaces)
        self.model = self.ModelCls(**self.env_model_kwargs,
            **self.model_kwargs)
        if share_memory:
            self.model.share_memory()
            # Store the shared_model (CPU) under a separate name, in case the
            # model gets moved to GPU later:
            self.shared_model = self.model
        if self.initial_model_state_dict is not None:
            self.model.load_state_dict(self.initial_model_state_dict)
        self.env_spaces = env_spaces
        self.share_memory = share_memory

    def make_env_to_model_kwargs(self, env_spaces):
        """Generate any keyword args to the model which depend on environment interfaces."""
        return {}

    def to_device(self, cuda_idx=None):
        """Moves the model to the specified cuda device, if not ``None``.  If
        sharing memory, instantiates a new model to preserve the shared (CPU)
        model.  Agents with additional model components (beyond
        ``self.model``) for action-selection or for use during training should
        extend this method to move those to the device, as well.

        Typically called in the runner during startup.
        """
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
        """Wraps the model with PyTorch's DistributedDataParallel.  The
        intention is for rlpyt to create a separate Python process to drive
        each GPU (or CPU-group for CPU-only, MPI-like configuration). Agents
        with additional model components (beyond ``self.model``) which will
        have gradients computed through them should extend this method to wrap
        those, as well.

        Typically called in the runner during startup.
        """
        device_id = self.device.index  # None if cpu, else cuda index.
        self.model = DDP(
            self.model,
            device_ids=None if device_id is None else [device_id],  # 1 GPU.
            output_device=device_id,
        )
        logger.log("Initialized DistributedDataParallel agent model on "
            f"device {self.device}.")
        return device_id

    def async_cpu(self, share_memory=True):
        """Used in async runner only; creates a new model instance to be used
        in the sampler, separate from the model shared with the optimizer
        process.  The sampler can operate asynchronously, and choose when to
        copy the optimizer's (shared) model parameters into its model (under
        read-write lock). The sampler model may be stored in shared memory,
        as well, to instantly share values with sampler workers.  Agents with
        additional model components (beyond ``self.model``) should extend this
        method to do the same with those, if using in asynchronous mode.

        Typically called in the runner during startup.

        TODO: double-check wording if this happens in sampler and optimizer."""
        if self.device.type != "cpu":
            return
        assert self.shared_model is not None
        self.model = self.ModelCls(**self.env_model_kwargs,
            **self.model_kwargs)
        # TODO: might need strip_ddp_state_dict.
        self.model.load_state_dict(self.shared_model.state_dict())
        if share_memory:  # Not needed in async_serial.
            self.model.share_memory()  # For CPU workers in async_cpu.
        logger.log("Initialized async CPU agent model.")

    def collector_initialize(self, global_B=1, env_ranks=None):
        """If needed to initialize within CPU sampler (e.g. vector epsilon-greedy,
        see EpsilonGreedyAgent for details)."""
        pass

    @torch.no_grad()  # Hint: apply this decorator on overriding method.
    def step(self, observation, prev_action, prev_reward):
        """Returns selected actions for environment instances in sampler."""
        raise NotImplementedError  # return type: AgentStep

    def reset(self):
        pass

    def reset_one(self, idx):
        pass

    def parameters(self):
        """Parameters to be optimized (overwrite in subclass if multiple models)."""
        return self.model.parameters()

    def state_dict(self):
        """Returns model parameters for saving."""
        return self.model.state_dict()

    def load_state_dict(self, state_dict):
        """Load model parameters, should expect format returned from ``state_dict()``."""
        self.model.load_state_dict(state_dict)

    def train_mode(self, itr):
        """Go into training mode (e.g. see PyTorch's ``Module.train()``)."""
        self.model.train()
        self._mode = "train"

    def sample_mode(self, itr):
        """Go into sampling mode."""
        self.model.eval()
        self._mode = "sample"

    def eval_mode(self, itr):
        """Go into evaluation mode.  Example use could be to adjust epsilon-greedy."""
        self.model.eval()
        self._mode = "eval"

    def sync_shared_memory(self):
        """Copies model parameters into shared_model, e.g. to make new values
        available to sampler workers.  If running CPU-only, these will be the
        same object--no copy necessary.  If model is on GPU, copy to CPU is
        performed. (Requires ``initialize(share_memory=True)`` called
        previously.  Not used in async mode.

        Typically called in the XXX during YY.
        """
        if self.shared_model is not self.model:  # (self.model gets trained)
            self.shared_model.load_state_dict(strip_ddp_state_dict(
                self.model.state_dict()))

    def send_shared_memory(self):
        """Used in async mode only, in optimizer process; copies parameters
        from trained model (maybe GPU) to shared model, which the sampler can
        access. Does so under write-lock, and increments send-count which sampler
        can check.

        Typically called in the XXX during YY."""
        if self.shared_model is not self.model:
            with self._rw_lock.write_lock:
                self.shared_model.load_state_dict(
                    strip_ddp_state_dict(self.model.state_dict()))
                self._send_count.value += 1

    def recv_shared_memory(self):
        """Used in async mode, in sampler process; copies parameters from
        model shared with optimizer into local model, if shared model has been
        updated.  Does so under read-lock.  (Local model might also be shared
        with sampler workers).

        Typically called in the XXX during YY."""
        if self.shared_model is not self.model:
            with self._rw_lock:
                if self._recv_count < self._send_count.value:
                    self.model.load_state_dict(self.shared_model.state_dict())
                    self._recv_count = self._send_count.value

    def toggle_alt(self):
        pass  # Only needed for recurrent alternating agent, but might get called.


AgentInputsRnn = namedarraytuple("AgentInputsRnn",  # Training only.
    ["observation", "prev_action", "prev_reward", "init_rnn_state"])


class RecurrentAgentMixin:
    """
    Mixin class to manage recurrent state during sampling (so the sampler
    remains agnostic).  To be used like ``class
    MyRecurrentAgent(RecurrentAgentMixin, MyAgent):``.
    """
    recurrent = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._prev_rnn_state = None
        self._sample_rnn_state = None  # Store during eval.

    def reset(self):
        """Sets the recurrent state to ``None``, which built-in PyTorch
        modules conver to zeros."""
        self._prev_rnn_state = None

    def reset_one(self, idx):
        """Sets the recurrent state corresponding to one environment instance
        to zero.  Assumes rnn state is in cudnn-compatible shape: [N,B,H],
        where B corresponds to environment index."""
        if self._prev_rnn_state is not None:
            self._prev_rnn_state[:, idx] = 0  # Automatic recursion in namedarraytuple.

    def advance_rnn_state(self, new_rnn_state):
        """Sets the recurrent state to the newly computed one (i.e. recurrent agents should
        call this at the end of their ``step()``). """
        self._prev_rnn_state = new_rnn_state

    @property
    def prev_rnn_state(self):
        return self._prev_rnn_state

    def train_mode(self, itr):
        """If coming from sample mode, store the rnn state elsewhere and clear it."""
        if self._mode == "sample":
            self._sample_rnn_state = self._prev_rnn_state
        self._prev_rnn_state = None
        super().train_mode(itr)

    def sample_mode(self, itr):
        """If coming from non-sample modes, restore the last sample-mode rnn state."""
        if self._mode != "sample":
            self._prev_rnn_state = self._sample_rnn_state
        super().sample_mode(itr)

    def eval_mode(self, itr):
        """If coming from sample mode, store the rnn state elsewhere and clear it."""
        if self._mode == "sample":
            self._sample_rnn_state = self._prev_rnn_state
        self._prev_rnn_state = None
        super().eval_mode(itr)


class AlternatingRecurrentAgentMixin:
    """
    Maintain an alternating pair of recurrent states to use when stepping in
    the sampler. Automatically swap them out when ``advance_rnn_state()`` is
    called, so it otherwise behaves like regular recurrent agent.  Should use
    only in alternating samplers, where two sets of environment instances take
    turns stepping (no special class needed for feedforward agents).  Use in
    place of ``RecurrentAgentMixin``.
    """

    recurrent = True
    alternating = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._alt = 0
        self._prev_rnn_state = None
        self._prev_rnn_state_pair = [None, None]
        self._sample_rnn_state_pair = [None, None]

    def reset(self):
        self._prev_rnn_state_pair = [None, None]
        self._prev_rnn_state = None
        self._alt = 0
        # Leave _sample_rnn_state_pair alone.

    def advance_rnn_state(self, new_rnn_state):
        """To be called inside agent.step()."""
        self._prev_rnn_state_pair[self._alt] = new_rnn_state
        self._alt ^= 1
        self._prev_rnn_state = self._prev_rnn_state_pair[self._alt]

    @property
    def prev_rnn_state(self):
        return self._prev_rnn_state

    def train_mode(self, itr):
        if self._mode == "sample":
            self._sample_rnn_state_pair = self._prev_rnn_state_pair
        self._prev_rnn_state_pair = [None, None]
        self._prev_rnn_state = None
        self._alt = 0
        super().train_mode(itr)

    def sample_mode(self, itr):
        if self._mode != "sample":
            self._prev_rnn_state_pair = self._sample_rnn_state_pair
            self._alt = 0
            self._prev_rnn_state = self._prev_rnn_state_pair[0]
        super().sample_mode(itr)

    def eval_mode(self, itr):
        if self._mode == "sample":
            self._sample_rnn_state_pair = self._prev_rnn_state_pair
        self._prev_rnn_state_pair = [None, None]
        self._prev_rnn_state = None
        self._alt = 0
        super().eval_mode(itr)

    def get_alt(self):
        return self._alt

    def toggle_alt(self):
        self._alt ^= 1
        self._prev_rnn_state = self._prev_rnn_state_pair[self._alt]
