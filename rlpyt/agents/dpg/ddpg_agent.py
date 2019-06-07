
import torch

from rlpyt.agents.base import BaseAgent, AgentStep
from rlpyt.utils.quick_args import save__init__args
from rlpyt.distributions.independent_gaussian import (IndependentGaussian,
    DistInfo)
from rlpyt.utils.buffer import buffer_to
from rlpyt.utils.logging import logger
from rlpyt.models.dpg.mlp import MlpMuModel, MlpQModel
from rlpyt.models.utils import update_state_dict
from rlpyt.utils.collections import namedarraytuple


AgentInfo = namedarraytuple("AgentInfo", ["mu"])


class DdpgAgent(BaseAgent):

    def __init__(self,
            MuModelCls=MlpMuModel,
            QModelCls=MlpQModel,
            mu_model_kwargs=None,
            q_model_kwargs=None,
            initial_mu_model_state_dict=None,
            initial_q_model_state_dict=None,
            ):
        if mu_model_kwargs is None:
            mu_model_kwargs = dict(hidden_sizes=[400, 300])
        if q_model_kwargs is None:
            q_model_kwargs = dict(hidden_sizes=[400, 300])
        save__init__args(locals())

    def initialize(self, env_spec, share_memory=False):
        env_model_kwargs = self.make_env_to_model_kwargs(env_spec)
        self.mu_model = self.MuModelCls(**env_model_kwargs, **self.mu_model_kwargs)
        self.q_model = self.QModelCls(**env_model_kwargs, **self.q_model_kwargs)
        if share_memory:
            self.mu_model.share_memory()
            # self.q_model.share_memory()  # Not needed for sampling.
            self.shared_mu_model = self.mu_model
            # self.shared_q_model = self.q_model
        if self.initial_mu_model_state_dict is not None:
            self.mu_model.load_state_dict(self.initial_mu_model_state_dict)
        if self.initial_q_model_state_dict is not None:
            self.q_model.load_state_dict(self.initial_q_model_state_dict)
        self.target_mu_model = self.MuModelCls(**env_model_kwargs,
            **self.mu_model_kwargs)
        self.target_mu_model.load_state_dict(self.mu_model.state_dict())
        self.target_q_model = self.QModelCls(**env_model_kwargs,
            **self.q_model_kwargs)
        self.target_q_model.load_state_dict(self.q_model.state_dict())
        self.distribution = IndependentGaussian(dim=env_spec.action_space.size)
        self.distribution.set_clip(env_spec.action_space.high)
        self.env_spec = env_spec
        self.env_model_kwargs = env_model_kwargs

    def initialize_cuda(self, cuda_idx=None):
        if cuda_idx is None:
            return  # CPU
        if self.shared_mu_model is not None:
            self.mu_model = self.MuModelCls(**self.env_model_kwargs,
                **self.mu_model_kwargs)
            self.mu_model.load_state_dict(self.shared_mu_model.state_dict())
        self.device = torch.device("cuda", index=cuda_idx)
        self.mu_model.to(self.device)
        self.q_model.to(self.device)
        self.target_mu_model.to(self.device)
        self.target_q_model.to(self.device)
        logger.log(f"Initialized agent models on device: {self.device}.")

    def make_env_to_model_kwargs(self, env_spec):
        return dict(
            observation_size=env_spec.observation_space.size,
            action_size=env_spec.action_space.size,
            obs_n_dim=len(env_spec.observation_space.shape),
        )

    def q(self, observation, prev_action, prev_reward, action):
        model_inputs = buffer_to((observation, prev_action, prev_reward,
            action), device=self.device)
        q = self.q_model(*model_inputs)
        return q.cpu()

    def q_at_mu(self, observation, prev_action, prev_reward):
        model_inputs = buffer_to((observation, prev_action, prev_reward),
            device=self.device)
        mu = self.mu_model(*model_inputs)
        q = self.q_model(*model_inputs, mu)
        return q.cpu()

    @torch.no_grad()
    def step(self, observation, prev_action, prev_reward):
        model_inputs = buffer_to((observation, prev_action, prev_reward),
            device=self.device)
        mu = self.mu_model(*model_inputs)
        action = self.distribution.sample(DistInfo(mean=mu))
        agent_info = AgentInfo(mu=mu)
        action, agent_info = buffer_to((action, agent_info), device="cpu")
        return AgentStep(action=action, agent_info=agent_info)

    @torch.no_grad()
    def target_q_at_mu(self, observation, prev_action, prev_reward):
        model_inputs = buffer_to((observation, prev_action, prev_reward),
            device=self.device)
        target_mu = self.target_mu_model(*model_inputs)
        target_q_at_mu = self.target_q_model(*model_inputs, target_mu)
        return target_q_at_mu.cpu()

    def update_target(self, tau=1):
        update_state_dict(self.target_mu_model, self.mu_model, tau)
        update_state_dict(self.target_q_model, self.q_model, tau)

    def sync_shared_memory(self):
        if self.shared_mu_model is not self.mu_model:
            self.shared_mu_model.load_state_dict(self.mu_model.state_dict())

    def q_parameters(self):
        return self.q_model.parameters()

    def mu_parameters(self):
        return self.mu_model.parameters()

    def set_policy_noise(self, std, noise_clip=None):
        self.distribution.set_std(std)
        self.distribution.set_noise_clip(noise_clip)
