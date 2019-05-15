
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

    bootstrap_value = True
    opt_info_fields = OptInfo._fields.copy()

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

    def initialize(self, agent, n_itr, mid_batch_reset=False):
        save_args(locals())
        self.optimizer = self.OptimCls(agent.parameters(),
            lr=self.learning_rate, **self.optim_kwargs)

    def loss(self, samples):
        dist_info, value = self.agent(samples)
        # TODO: try to compute everyone on device.
        return_, advantage, valid = self.process_samples(samples)

        dist = self.agent.distribution
        logli = dist.log_likelihood(samples.agent.action, dist_info)
        pi_loss = - valid_mean(logli * advantage, valid)

        value_error = 0.5 * (value - return_) ** 2
        value_loss = self.value_loss_coeff * valid_mean(value_error, valid)

        entropy = dist.mean_entropy(dist_info, valid)
        entropy_loss = - self.entropy_loss_coeff * entropy

        loss = pi_loss + value_loss + entropy_loss

        perplexity = dist.mean_perplexity(dist_info, valid)

        return loss, entropy, perplexity, OptData(return_, advantage, valid)

    def optimize_agent(self, train_samples, itr):
        self.optimizer.zero_grad()
        loss, entropy, perplexity, opt_data = self.loss(train_samples)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.agent.model.parameters(), self.clip_grad_norm)
        self.optimizer.step()
        opt_info = OptInfo(i.item()
            for i in [loss, grad_norm, entropy, perplexity])
        return opt_data, opt_info

    def process_samples(self, samples):
        return_ = discount_return(samples.env.reward, samples.env.done,
            samples.agent.bootstrap_value, self.discount)
        advantage = return_ - samples.agent.value
        if self.mid_batch_reset:
            valid = torch.ones_like(samples.env.done)
        else:
            valid = 1 - torch.clamp(torch.cumsum(samples.env.done, dim=0),
                max=1)
        return return_, advantage, valid
