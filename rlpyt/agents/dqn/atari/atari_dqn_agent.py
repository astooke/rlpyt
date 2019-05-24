
import torch

from rlpyt.agents.base import BaseAgent, AgentStep
from rlpyt.agents.q_learning.base import AgentInfo
from rlpyt.models.q_learning.atari_dqn_model import AtariDqnModel
from rlpyt.distributions import EpsilonGreedy
from rlpyt.utils.buffer import buffer_to


class AtariDqnAgent(BaseAgent):

    def __init__(self, ModelCls=AtariDqnModel, **kwargs):
        super().__init__(ModelCls=ModelCls, **kwargs)

    def __call__(self, observation, prev_action, prev_reward):
        model_inputs = buffer_to((observation, prev_action, prev_reward),
            device=self.device)
        q = self.model(*model_inputs)
        return q.cpu()

    def initialize(self, env_spec, share_memory=False):
        super().initialize(env_spec, share_memory)
        self.distribution = EpsilonGreedy()

    def make_env_to_model_kwargs(self, env_spec):
        return dict(image_shape=env_spec.observation_space.shape,
                    output_dim=env_spec.action_space.n)

    @torch.no_grad()
    def step(self, observation, prev_action, prev_reward):
        model_inputs = buffer_to((observation, prev_action, prev_reward),
            device=self.device)
        q = self.model(*model_inputs)
        action = self.distribution.sample(q)
        agent_info = AgentInfo(q=q)
        action, agent_info = buffer_to((action, agent_info), device="cpu")
        return AgentStep(action=action, agent_info=agent_info)

    @torch.no_grad()
    def target_q(self, observation, prev_action, prev_reward):
        model_inputs = buffer_to((observation, prev_action, prev_reward),
            device=self.device())
        target_q = self.target_model(*model_inputs)
        return target_q.cpu()

    def set_epsilon_greedy(self, epsilon):
        self.sample_epsilon = epsilon
        self.distribution.set_epsilon(epsilon)

    def give_eval_epsilon_greedy(self, epsilon):
        self.eval_epsilon = epsilon

    def train_mode(self):
        self.model.train()

    def sample_mode(self):
        self.model.eval()
        self.distribution.set_epsilon(self.sample_epsilon)

    def eval_mode(self):
        self.model.eval()
        self.distribution.set_epsilon(self.eval_epsilon)

    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())
