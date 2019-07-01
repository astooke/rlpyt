import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from rlpyt.utils.collections import namedarraytuple
from rlpyt.agents.base import BaseAgent
from rlpyt.utils.logging import logger

AgentInfo = namedarraytuple("AgentInfo", ["dist_info", "value"])
AgentInfoRnn = namedarraytuple("AgentInfoRnn",
    ["dist_info", "value", "prev_rnn_state"])


class BasePgAgent(BaseAgent):

    distribution = None  # type: Distribution

    def initialize(self, env_spaces, share_memory=False):
        env_model_kwargs = self.make_env_to_model_kwargs(env_spaces)
        self.model = self.ModelCls(**env_model_kwargs, **self.model_kwargs)
        if share_memory:
            self.model.share_memory()
            self.shared_model = self.model
        if self.initial_model_state_dict is not None:
            self.model.load_state_dict(self.initial_model_state_dict)
        self.env_spaces = env_spaces
        self.env_model_kwargs = env_model_kwargs
        self.share_memory = share_memory

    def initialize_cuda(self, cuda_idx=None, ddp=False):
        if cuda_idx is None:
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

    def make_env_to_model_kwargs(self, env_spaces):
        raise NotImplementedError
