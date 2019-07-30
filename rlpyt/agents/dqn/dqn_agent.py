
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.parallel import DistributedDataParallelCPU as DDPC

from rlpyt.agents.base import BaseAgent, AgentStep
from rlpyt.agents.dqn.epsilon_greedy import EpsilonGreedyAgentMixin
from rlpyt.distributions.epsilon_greedy import EpsilonGreedy
from rlpyt.utils.buffer import buffer_to
from rlpyt.utils.logging import logger
from rlpyt.utils.collections import namedarraytuple


AgentInfo = namedarraytuple("AgentInfo", "q")


class DqnAgent(EpsilonGreedyAgentMixin, BaseAgent):

    def __call__(self, observation, prev_action, prev_reward):
        prev_action = self.distribution.to_onehot(prev_action)
        model_inputs = buffer_to((observation, prev_action, prev_reward),
            device=self.device)
        q = self.model(*model_inputs)
        return q.cpu()

    def initialize(self, env_spaces, share_memory=False,
            global_B=1, env_ranks=None):
        env_model_kwargs = self.make_env_to_model_kwargs(env_spaces)
        self.model = self.ModelCls(**env_model_kwargs, **self.model_kwargs)
        if share_memory:
            self.model.share_memory()
            self.shared_model = self.model
        if self.initial_model_state_dict is not None:
            self.model.load_state_dict(self.initial_model_state_dict)
        self.target_model = self.ModelCls(**env_model_kwargs, **self.model_kwargs)
        self.target_model.load_state_dict(self.model.state_dict())
        self.distribution = EpsilonGreedy(dim=env_spaces.action.n)
        self.env_spaces = env_spaces
        self.env_model_kwargs = env_model_kwargs
        self.share_memory = share_memory
        if env_ranks is not None:
            self.make_vec_eps(global_B, env_ranks)

    def initialize_device(self, cuda_idx=None, ddp=False):
        if cuda_idx is None:
            if ddp:
                self.model = DDPC(self.model)
                logger.log("Initialized DistributedDataParallelCPU agent model.")
            return  # CPU
        if self.shared_model is not None:
            self.model = self.ModelCls(**self.env_model_kwargs,
                **self.model_kwargs)
            self.model.load_state_dict(self.shared_model.state_dict())
        self.device = torch.device("cuda", index=cuda_idx)
        self.model.to(self.device)
        if ddp:
            self.model = DDP(self.model, device_ids=[cuda_idx],
                output_device=cuda_idx)
            logger.log("Initialized DistributedDataParallel agent model "
                f"on device: {self.device}.")
        else:
            logger.log(f"Initialized agent model on device: {self.device}.")
        self.target_model.to(self.device)

    def make_env_to_model_kwargs(self, env_spaces):
        raise NotImplementedError

    def state_dict(self):
        return dict(model=self.model.state_dict(),
            target=self.target_model.state_dict())

    @torch.no_grad()
    def step(self, observation, prev_action, prev_reward):
        prev_action = self.distribution.to_onehot(prev_action)
        model_inputs = buffer_to((observation, prev_action, prev_reward),
            device=self.device)
        q = self.model(*model_inputs)
        q = q.cpu()
        action = self.distribution.sample(q)
        agent_info = AgentInfo(q=q)
        # action, agent_info = buffer_to((action, agent_info), device="cpu")
        return AgentStep(action=action, agent_info=agent_info)

    def target(self, observation, prev_action, prev_reward):
        prev_action = self.distribution.to_onehot(prev_action)
        model_inputs = buffer_to((observation, prev_action, prev_reward),
            device=self.device)
        target_q = self.target_model(*model_inputs)
        return target_q.cpu()

    def update_target(self):
        # Workaround the fact that DistributedDataParallel prepends 'module.' to
        # every key, but the target model will not be wrapped in
        # DistributedDataParallel.
        # (Solution from PyTorch forums.)
        model_state_dict = self.model.state_dict()
        new_state_dict = type(model_state_dict)()
        for k, v in model_state_dict.items():
            if k[:7] == "module.":
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        self.target_model.load_state_dict(new_state_dict)
