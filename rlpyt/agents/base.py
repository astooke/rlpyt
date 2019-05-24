
import torch

from rlpyt.utils.quick_args import save__init__args
from rlpyt.utils.collections import namedarraytuple

AgentInputs = namedarraytuple("AgentStepInputs",
    ["observation", "prev_action", "prev_reward"])
AgentStep = namedarraytuple("AgentStep", ["action", "agent_info"])


class BaseAgent(object):

    model = None  # type: torch.nn.Module
    shared_model = None
    device = torch.device("cpu")
    recurrent = False

    def __init__(self, ModelCls, model_kwargs, initial_model_state_dict=None):
        save__init__args(locals())
        self.env_model_kwargs = dict()  # Populate in initialize().

    def __call__(self, observation, prev_action, prev_reward):
        """Returns values from model forward pass on training data."""
        raise NotImplementedError

    def initialize(self, env_spec, share_memory=False):
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
        """Parameters to be optimized."""
        return self.model.parameters()

    def train_mode(self):
        """Go into training mode."""
        self.model.train()

    def sample_mode(self):
        """Go into sampling mode."""
        self.model.eval()

    def eval_mode(self):
        """Go into evaluation mode."""
        self.model.eval()

    def sync_shared_memory(self):
        """Call in sampler master, after share_memory=True to initialize()."""
        if self.shared_model is not self.model:  # (self.model gets trained)
            self.shared_model.load_state_dict(self.model.state_dict())
