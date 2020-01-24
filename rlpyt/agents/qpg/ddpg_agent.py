
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.parallel import DistributedDataParallelCPU as DDPC

from rlpyt.agents.base import BaseAgent, AgentStep
from rlpyt.utils.quick_args import save__init__args
from rlpyt.distributions.gaussian import Gaussian, DistInfo
from rlpyt.utils.buffer import buffer_to
from rlpyt.utils.logging import logger
from rlpyt.models.qpg.mlp import MuMlpModel, QofMuMlpModel
from rlpyt.models.utils import update_state_dict
from rlpyt.utils.collections import namedarraytuple


AgentInfo = namedarraytuple("AgentInfo", ["mu"])


class DdpgAgent(BaseAgent):
    """Agent for deep deterministic policy gradient algorithm."""

    shared_mu_model = None

    def __init__(
            self,
            ModelCls=MuMlpModel,  # Mu model.
            QModelCls=QofMuMlpModel,
            model_kwargs=None,  # Mu model.
            q_model_kwargs=None,
            initial_model_state_dict=None,  # Mu model.
            initial_q_model_state_dict=None,
            action_std=0.1,
            action_noise_clip=None,
            ):
        """Saves input arguments; default network sizes saved here."""
        if model_kwargs is None:
            model_kwargs = dict(hidden_sizes=[400, 300])
        if q_model_kwargs is None:
            q_model_kwargs = dict(hidden_sizes=[400, 300])
        save__init__args(locals())
        super().__init__()  # For async setup.

    def initialize(self, env_spaces, share_memory=False,
            global_B=1, env_ranks=None):
        """Instantiates mu and q, and target_mu and target_q models."""
        super().initialize(env_spaces, share_memory,
            global_B=global_B, env_ranks=env_ranks)
        self.q_model = self.QModelCls(**self.env_model_kwargs,
            **self.q_model_kwargs)
        if self.initial_q_model_state_dict is not None:
            self.q_model.load_state_dict(self.initial_q_model_state_dict)
        self.target_model = self.ModelCls(**self.env_model_kwargs,
            **self.model_kwargs)
        self.target_q_model = self.QModelCls(**self.env_model_kwargs,
            **self.q_model_kwargs)
        self.target_q_model.load_state_dict(self.q_model.state_dict())
        assert len(env_spaces.action.shape) == 1
        self.distribution = Gaussian(
            dim=env_spaces.action.shape[0],
            std=self.action_std,
            noise_clip=self.action_noise_clip,
            clip=env_spaces.action.high[0],  # Assume symmetric low=-high.
        )

    def to_device(self, cuda_idx=None):
        super().to_device(cuda_idx)  # Takes care of self.model.
        self.target_model.to(self.device)
        self.q_model.to(self.device)
        self.target_q_model.to(self.device)

    def data_parallel(self):
        super().data_parallel()  # Takes care of self.model.
        if self.device.type == "cpu":
            self.q_model = DDPC(self.q_model)
        else:
            self.q_model = DDP(self.q_model)

    def make_env_to_model_kwargs(self, env_spaces):
        assert len(env_spaces.action.shape) == 1
        return dict(
            observation_shape=env_spaces.observation.shape,
            action_size=env_spaces.action.shape[0],
        )

    def q(self, observation, prev_action, prev_reward, action):
        """Compute Q-value for input state/observation and action (with grad)."""
        model_inputs = buffer_to((observation, prev_action, prev_reward,
            action), device=self.device)
        q = self.q_model(*model_inputs)
        return q.cpu()

    def q_at_mu(self, observation, prev_action, prev_reward):
        """Compute Q-value for input state/observation, through the mu_model
        (with grad)."""
        model_inputs = buffer_to((observation, prev_action, prev_reward),
            device=self.device)
        mu = self.model(*model_inputs)
        q = self.q_model(*model_inputs, mu)
        return q.cpu()

    def target_q_at_mu(self, observation, prev_action, prev_reward):
        """Compute target Q-value for input state/observation, through the
        target mu_model."""
        model_inputs = buffer_to((observation, prev_action, prev_reward),
            device=self.device)
        target_mu = self.target_model(*model_inputs)
        target_q_at_mu = self.target_q_model(*model_inputs, target_mu)
        return target_q_at_mu.cpu()

    @torch.no_grad()
    def step(self, observation, prev_action, prev_reward):
        """Computes distribution parameters (mu) for state/observation,
        returns (gaussian) sampled action."""
        model_inputs = buffer_to((observation, prev_action, prev_reward),
            device=self.device)
        mu = self.model(*model_inputs)
        action = self.distribution.sample(DistInfo(mean=mu))
        agent_info = AgentInfo(mu=mu)
        action, agent_info = buffer_to((action, agent_info), device="cpu")
        return AgentStep(action=action, agent_info=agent_info)

    def update_target(self, tau=1):
        update_state_dict(self.target_model, self.model.state_dict(), tau)
        update_state_dict(self.target_q_model, self.q_model.state_dict(), tau)

    def q_parameters(self):
        return self.q_model.parameters()

    def mu_parameters(self):
        return self.model.parameters()

    def train_mode(self, itr):
        super().train_mode(itr)
        self.q_model.train()

    def sample_mode(self, itr):
        super().sample_mode(itr)
        self.q_model.eval()
        self.distribution.set_std(self.action_std)

    def eval_mode(self, itr):
        super().eval_mode(itr)
        self.q_model.eval()
        self.distribution.set_std(0.)  # Deterministic.

    def state_dict(self):
        return dict(
            model=self.model.state_dict(),
            q_model=self.q_model.state_dict(),
            target_model=self.target_model.state_dict(),
            target_q_model=self.target_q_model.state_dict(),
        )

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict["model"])
        self.q_model.load_state_dict(state_dict["q_model"])
        self.target_model.load_state_dict(state_dict["target_model"])
        self.target_q_model.load_state_dict(state_dict["target_q_model"])
