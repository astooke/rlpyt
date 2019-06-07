
import torch

from rlpyt.agents.base import AgentStep
from rlpyt.agents.q_learning.epsilon_greedy import EpsilonGreedyAgent
from rlpyt.distributions.epislon_greedy import CatDistInfo
from rlpyt.models.q_learning.atari_catdqn_model import AtariCatDqnModel
from rlpyt.utils.buffer import buffer_to
from rlpyt.utils.collections import namedarraytuple
from rlpyt.utils.logging import logger

AgentInfo = namedarraytuple("AgentInfo", ["p"])


class AtariCatDqnAgent(EpsilonGreedyAgent):

    def __init__(self, ModelCls=AtariCatDqnModel, n_atoms=51, **kwargs):
        super().__init__(ModelCls=ModelCls, n_atoms=n_atoms, **kwargs)
        self.model_kwargs["n_atoms"] = n_atoms

    def __call__(self, observation, prev_action, prev_reward):
        model_inputs = buffer_to((observation, prev_action, prev_reward),
            device=self.device)
        p = self.model(*model_inputs)
        return p.cpu()

    def give_V_min_max(self, V_min, V_max):
        self.V_min = V_min
        self.V_max = V_max
        self.distribution.give_z(torch.linspace(V_min, V_max, self.n_atoms))

    def initialize(self, env_spec, share_memory=False):
        env_model_kwargs = self.make_env_to_model_kwargs(env_spec)
        self.model = self.ModelCls(**env_model_kwargs, **self.model_kwargs)
        if share_memory:
            self.model.share_memory()
            self.shared_model = self.model
        if self.initial_model_state_dict is not None:
            self.model.load_state_dict(self.initial_model_state_dict)
        self.target_model = self.ModelCls(**env_model_kwargs, **self.model_kwargs)
        self.target_model.load_state_dict(self.model.state_dict())
        self.env_spec = env_spec
        self.env_model_kwargs = env_model_kwargs
        super().initialize(env_spec, share_memory)

    def initialize_cuda(self, cuda_idx=None):
        if cuda_idx is None:
            return  # CPU
        if self.shared_model is not None:
            self.model = self.ModelCls(**self.env_model_kwargs,
                **self.model_kwargs)
            self.model.load_state_dict(self.shared_model.state_dict())
        self.device = torch.device("cuda", index=cuda_idx)
        self.model.to(self.device)
        self.target_model.to(self.device)
        logger.log(f"Initialized agent model on device: {self.device}.")

    def make_env_to_model_kwargs(self, env_spec):
        return dict(image_shape=env_spec.observation_space.shape,
                    output_dim=env_spec.action_space.n)

    @torch.no_grad()
    def step(self, observation, prev_action, prev_reward):
        model_inputs = buffer_to((observation, prev_action, prev_reward),
            device=self.device)
        p = self.model(*model_inputs)
        action = self.distribution.sample(CatDistInfo(p=p))
        agent_info = AgentInfo(p=p)
        action, agent_info = buffer_to((action, agent_info), device="cpu")
        return AgentStep(action=action, agent_info=agent_info)

    @torch.no_grad()
    def target_p(self, observation, prev_action, prev_reward):
        model_inputs = buffer_to((observation, prev_action, prev_reward),
            device=self.device())
        target_p = self.target_model(*model_inputs)
        return target_p.cpu()

    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())
