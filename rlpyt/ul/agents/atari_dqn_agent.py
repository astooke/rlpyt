
import torch
import copy
from collections import OrderedDict

from rlpyt.agents.base import BaseAgent, AgentStep
from rlpyt.utils.quick_args import save__init__args
from rlpyt.agents.dqn.epsilon_greedy import EpsilonGreedyAgentMixin
from rlpyt.distributions.epsilon_greedy import EpsilonGreedy
from rlpyt.utils.buffer import buffer_to
from rlpyt.utils.logging import logger
from rlpyt.utils.collections import namedarraytuple
from rlpyt.models.utils import update_state_dict

from rlpyt.ul.models.rl.atari_rl_models import AtariDqnModel


AgentInfoConv = namedarraytuple("AgentInfo", ["q", "conv"])


class AtariDqnAgent(EpsilonGreedyAgentMixin, BaseAgent):
    """
    Standard agent for DQN algorithms with epsilon-greedy exploration.  
    """

    def __init__(
            self,
            ModelCls=AtariDqnModel,
            model_kwargs=None,
            load_conv=False,
            load_all=False,
            state_dict_filename=None,
            store_latent=False,
            **kwargs
            ):
        if model_kwargs is None:
            model_kwargs = dict()
        assert not (load_conv and load_all)
        save__init__args(locals())
        super().__init__(ModelCls=ModelCls, **kwargs)

    def __call__(self, observation, prev_action, prev_reward):
        """Returns Q-values for states/observations (with grad)."""
        prev_action = self.distribution.to_onehot(prev_action)
        model_inputs = buffer_to((observation, prev_action, prev_reward),
            device=self.device)
        q, _conv = self.model(*model_inputs)
        return q.cpu() 

    def initialize(self, env_spaces, share_memory=False,
            global_B=1, env_ranks=None):
        """Along with standard initialization, creates vector-valued epsilon
        for exploration, if applicable, with a different epsilon for each
        environment instance."""
        self.model = self.ModelCls(
            image_shape=env_spaces.observation.shape,
            action_size=env_spaces.action.n,
            **self.model_kwargs
        )
        if self.load_conv:
            logger.log("Agent loading state dict: " + self.state_dict_filename)
            loaded_state_dict = torch.load(self.state_dict_filename,
                map_location=torch.device('cpu'))
            # From UL, saves snapshot: params["algo_state_dict"]["encoder"]
            loaded_state_dict = loaded_state_dict.get("algo_state_dict", loaded_state_dict)
            loaded_state_dict = loaded_state_dict.get("encoder", loaded_state_dict)
            # A bit onerous, but ensures that state dicts match:
            conv_state_dict = OrderedDict([(k.replace("conv.", "", 1), v)
                for k, v in loaded_state_dict.items() if k.startswith("conv.")])
            self.model.conv.load_state_dict(conv_state_dict)
            logger.log("Agent loaded CONV state dict.")
        elif self.load_all:
            # From RL, saves snapshot: params["agent_state_dict"]
            loaded_state_dict = torch.load(self.state_dict_filename,
                map_location=torch.device('cpu'))
            self.load_state_dict(loaded_state_dict["agent_state_dict"])
            logger.log("Agnet loaded FULL state dict.")
        else:
            logger.log("Agent NOT loading state dict.")

        self.target_model = copy.deepcopy(self.model)
        self.distribution = EpsilonGreedy(dim=env_spaces.action.n)
        if env_ranks is not None:
            self.make_vec_eps(global_B, env_ranks)
        if share_memory:
            self.model.share_memory()
            self.shared_model = self.model
        if self.initial_model_state_dict is not None:
            raise NotImplementedError
        self.env_spaces = env_spaces
        self.share_memory = share_memory

    def to_device(self, cuda_idx=None):
        super().to_device(cuda_idx)
        self.target_model.to(self.device)

    def state_dict(self):
        return dict(model=self.model.state_dict(),
            target=self.target_model.state_dict())

    @torch.no_grad()
    def step(self, observation, prev_action, prev_reward):
        """Computes Q-values for states/observations and selects actions by
        epsilon-greedy. (no grad)"""
        prev_action = self.distribution.to_onehot(prev_action)
        model_inputs = buffer_to((observation, prev_action, prev_reward),
            device=self.device)
        q, conv = self.model(*model_inputs)
        q = q.cpu()
        action = self.distribution.sample(q)
        agent_info = AgentInfoConv(q=q,
            conv=conv if self.store_latent else None)
        # action, agent_info = buffer_to((action, agent_info), device="cpu")
        return AgentStep(action=action, agent_info=agent_info)

    def target(self, observation, prev_action, prev_reward):
        """Returns the target Q-values for states/observations."""
        prev_action = self.distribution.to_onehot(prev_action)
        model_inputs = buffer_to((observation, prev_action, prev_reward),
            device=self.device)
        target_q, _conv = self.target_model(*model_inputs)
        return target_q.cpu()

    def update_target(self, tau=1):
        """Copies the model parameters into the target model."""
        update_state_dict(self.target_model, self.model.state_dict(), tau)
