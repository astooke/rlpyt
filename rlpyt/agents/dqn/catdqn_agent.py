
import torch

from rlpyt.agents.base import AgentStep
from rlpyt.agents.dqn.dqn_agent import DqnAgent
from rlpyt.distributions.epsilon_greedy import CategoricalEpsilonGreedy
from rlpyt.utils.buffer import buffer_to
from rlpyt.utils.collections import namedarraytuple

AgentInfo = namedarraytuple("AgentInfo", ["p"])


class CatDqnAgent(DqnAgent):

    def __init__(self, n_atoms=51, **kwargs):
        super().__init__(**kwargs)
        self.model_kwargs["n_atoms"] = n_atoms

    def initialize(self, env_spec, share_memory=False):
        super().initialize(env_spec, share_memory)  # Then overwrite distribution.
        self.distribution = CategoricalEpsilonGreedy(dim=env_spec.action_space.n)

    def give_V_min_max(self, V_min, V_max):
        self.V_min = V_min
        self.V_max = V_max
        self.distribution.give_z(torch.linspace(V_min, V_max, self.n_atoms))

    @torch.no_grad()
    def step(self, observation, prev_action, prev_reward):
        prev_action = self.distribution.to_onehot(prev_action)
        model_inputs = buffer_to((observation, prev_action, prev_reward),
            device=self.device)
        p = self.model(*model_inputs)
        p = p.cpu()
        action = self.distribution.sample(p)
        agent_info = AgentInfo(p=p)  # Only change from DQN: q -> p.
        action, agent_info = buffer_to((action, agent_info), device="cpu")
        return AgentStep(action=action, agent_info=agent_info)
