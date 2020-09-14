
import torch
from collections import OrderedDict

from rlpyt.agents.base import BaseAgent, AgentStep
from rlpyt.ul.models.rl.atari_rl_models import AtariPgModel
from rlpyt.utils.buffer import buffer_to
from rlpyt.distributions.categorical import Categorical, DistInfo
from rlpyt.agents.pg.base import AgentInfo
from rlpyt.utils.collections import namedarraytuple
from rlpyt.utils.logging import logger

AgentInfoConv = namedarraytuple("AgentInfoConv", AgentInfo._fields + ("conv",))


class AtariPgAgent(BaseAgent):
    """Only doing the feedforward agent for now."""

    def __init__(
            self,
            ModelCls=AtariPgModel,
            store_latent=False,
            state_dict_filename=None,
            load_conv=False,
            load_all=False,
            **kwargs
        ):
        super().__init__(ModelCls=ModelCls, **kwargs)
        self.store_latent = store_latent
        self.state_dict_filename = state_dict_filename
        self.load_conv = load_conv
        self.load_all = load_all
        assert not (load_all and load_conv)
        self._act_uniform = False

    def __call__(self, observation, prev_action, prev_reward):
        prev_action = self.distribution.to_onehot(prev_action)
        model_inputs = buffer_to((observation, prev_action, prev_reward),
            device=self.device)
        pi, value, _ = self.model(*model_inputs)  # ignore conv output
        return buffer_to((DistInfo(prob=pi), value), device="cpu")

    def initialize(self, env_spaces, share_memory=False,
            global_B=1, env_ranks=None):
        self.model = self.ModelCls(
            image_shape=env_spaces.observation.shape,
            action_size=env_spaces.action.n,
            **self.model_kwargs
        )  # Model will have stop_grad inside it.
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

        if share_memory:
            self.model.share_memory()
            self.shared_model = self.model
        if self.initial_model_state_dict is not None:
            raise NotImplementedError
        self.distribution = Categorical(dim=env_spaces.action.n)
        self.env_spaces = env_spaces
        self.share_memory = share_memory

    @torch.no_grad()
    def step(self, observation, prev_action, prev_reward):
        prev_action = self.distribution.to_onehot(prev_action)
        model_inputs = buffer_to((observation, prev_action, prev_reward),
            device=self.device)
        pi, value, conv = self.model(*model_inputs)
        if self._act_uniform:
            pi[:] = 1. / pi.shape[-1]  # uniform
        dist_info = DistInfo(prob=pi)
        action = self.distribution.sample(dist_info)
        agent_info = AgentInfoConv(dist_info=dist_info, value=value,
            conv=conv if self.store_latent else None)  # Don't write extra data.
        action, agent_info = buffer_to((action, agent_info), device="cpu")
        return AgentStep(action=action, agent_info=agent_info)

    @torch.no_grad()
    def value(self, observation, prev_action, prev_reward):
        prev_action = self.distribution.to_onehot(prev_action)
        model_inputs = buffer_to((observation, prev_action, prev_reward),
            device=self.device)
        _pi, value, _ = self.model(*model_inputs)  # Ignore conv out
        return value.to("cpu")

    def set_act_uniform(self, act_uniform=True):
        self._act_uniform = act_uniform
