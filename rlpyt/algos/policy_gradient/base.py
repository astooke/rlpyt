
import torch
from collections import namedtuple

from rlpyt.algos.base import RlAlgorithm
from rlpyt.utils.collections import namedarraytuple
from rlpyt.algos.utils import discount_return, generalized_advantage_estimation

OptData = namedarraytuple("OptData", ["return_", "advantage", "valid"])
# Convention: traj_info fields CamelCase, opt_info fields lowerCamelCase
OptInfo = namedtuple("OptInfo", ["loss", "gradNorm", "entropy", "perplexity"])
AgentTrain = namedtuple("AgentTrain", ["dist_info", "value"])


class PolicyGradient(RlAlgorithm):

    bootstrap_value = True
    opt_info_fields = tuple(f for f in OptInfo._fields)  # copy

    def initialize(self, agent, n_itr, mid_batch_reset=False):
        self.optimizer = self.OptimCls(agent.parameters(),
            lr=self.learning_rate, **self.optim_kwargs)
        if self.initial_optim_state_dict is not None:
            self.optimizer.load_state_dict(self.initial_optim_state_dict)
        self.agent = agent
        self.n_itr = n_itr
        self.mid_batch_reset = mid_batch_reset

    def process_samples(self, samples):
        reward, done, value, bv = (samples.env.reward, samples.env.done,
            samples.agent.agent_info.value, samples.agent.bootstrap_value)
        done = done.type(reward.dtype)
        if self.gae_lambda == 1:  # GAE reduces to empirical discounted.
            return_ = discount_return(reward, done, bv, self.discount)
            advantage = return_ - value
        else:
            advantage, return_ = generalized_advantage_estimation(
                reward, value, done, bv, self.discount, self.gae_lambda)
        
        valid = torch.ones_like(done)  # or None
        if not self.mid_batch_reset:  # valid until 1 after first done
            valid[1:] = 1 - torch.clamp(torch.cumsum(done[:-1], dim=0), max=1)
        return return_, advantage, valid
