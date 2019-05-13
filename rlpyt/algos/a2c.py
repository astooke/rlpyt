
import numpy as np
import torch
from collections import namedtuple

from rlpyt.algos.base import RlAlgorithm
from rlpyt.utils.quick_args import save_args
from rlpyt.algos.utils import discount_return
from rlpyt.utils.tensor import valid_mean
from rlpyt.utils.collections import namedarraytuple


OptData = namedarraytuple("OptData", ["return", "advantage", "valid"])
OptInfo = namedtuple("OptInfo", ["Loss", "GradNorm", "Entropy", "Perplexity"])


class A2C(RlAlgorithm):

    opt_info_fields = OptInfo._fields

    def __init__(
            self,
            discount=0.99,
            learning_rate=0.001,
            value_loss_coeff=0.5,
            entropy_loss_coeff=0.01,
            OptimCls=torch.optim.Adam,
            optim_kwargs=None,
            clip_grad_norm=1.,
            ):
        if optim_kwargs is None:
            optim_kwargs = dict()
        save_args(locals())

    def initialize(self, agent, n_itr):
        save_args(locals())
        self.optimizer = self.OptimCls(agent.parameters(),
            lr=self.learning_rate, **self.optim_kwargs)

    def loss(self, agent_samples, env_samples):
        agent_info = self.agent(agent_samples, env_samples)
        return_, advantage, valid = self.process_samples(agent_samples, env_samples)

        dist = self.agent.distribution
        pi_logli = dist.log_likelihood(agent_samples.actions, agent_info.dist_info)
        pi_loss = - valid_mean(pi_logli * advantage, valid)

        value_error = 0.5 * (agent_info.value - return_) ** 2
        value_loss = self.value_loss_coeff * valid_mean(value_error, valid)

        entropy = dist.mean_entropy(agent_info.dist_info, valid)
        entropy_loss = - self.entropy_loss_coeff * entropy

        loss = pi_loss + value_loss + entropy_loss

        perplexity = dist.mean_perplexity(agent_info.dist_info, valid)

        return loss, entropy, perplexity, OptData(return_, advantage, valid)

    def optimize_agent(self, samples, itr):
        self.optimizer.zero_grad()
        loss, entropy, perplexity, opt_data = self.loss(samples)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.agent.parameters(),
            self.clip_grad_norm)
        self.optimizer.step()
        opt_info = OptInfo(i.item()
            for i in [loss, grad_norm, entropy, perplexity])
        return opt_data, opt_info

    def process_samples(self, samples):
        return_ = discount_return(samples.env.reward, samples.env.done,
            samples.agent.bootstrap_value, self.discount)
        advantage = return_ - samples.agent.value
        if self.mid_batch_reset:
            valid = np.ones_like(samples.env.done)
        else:
            valid = 1 - np.minimum(np.cumsum(samples.env.done, axis=0), 1)
        return return_, advantage, valid
