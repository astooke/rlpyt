
from rlpyt.utils.collections import namedarraytuple
from rlpyt.agents.base import BaseAgent

AgentInfo = namedarraytuple("AgentInfo", ["dist_info", "value"])


class BasePgAgent(BaseAgent):

    distribution = None  # type: Distribution

    def initialize(self, env_spec, share_memory=False):
        env_model_kwargs = self.make_env_to_model_kwargs(env_spec)
        self.model = self.ModelCls(**env_model_kwargs, **self.model_kwargs)
        if share_memory:
            self.model.share_memory()
            self.shared_model = self.model
        if self.initial_model_state_dict is not None:
            self.model.load_state_dict(self.initial_model_state_dict)
        self.env_spec = env_spec
        self.env_model_kwargs = env_model_kwargs

    def initialize_cuda(self, cuda_idx=None):
        if cuda_idx is None:
            return  # CPU
        if self.shared_model is not None:
            self.model = self.ModelCls(**self.env_model_kwargs, 
                **self.model_kwargs)
            self.model.load_state_dict(self.shared_model.state_dict())
        self.device = torch.device("cuda", index=cuda_idx)
        self.model.to(device)
        logger.log(f"Initialized agent model on device: {self.device}.")


    def make_env_to_model_kwargs(self, env_spec):
        return {}
