
import torch

from rlpyt.uitls.collections import namedarraytuple
from rlpyt.agents.base import BaseAgent, BaseRecurrentAgent


AgentInfo = namedarraytuple("AgentInfo", ["dist_info", "value"])


class BasePgAgent(BaseAgent):

    distribution = None  # type: Distribution

    def initialize(self, env_spec, share_memory=False):
        self._env_spec = env_spec
        self.model = self._ModelCls(env_spec, **self._model_kwargs)
        if share_memory:
            self.model.share_memory()
            self._shared_memory = share_memory
        if self._initial_state_dict is not None:
            self.model.load_state_dict(self._initial_state_dict)

    @torch.no_grad()  # Hint: apply this decorator on overriding method.
    def step(self, observation, prev_action, prev_reward):
        raise NotImplementedError  # return types: action, AgentInfo


RecurrentAgentInfo = namedarraytuple("AgentInfo",
    ["dist_info", "value", "prev_rnn_state"])


class BaseRecurrentPgAgent(BasePgAgent, BaseRecurrentAgent):

    pass
