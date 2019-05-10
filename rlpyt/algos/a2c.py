
import numpy as np
import torch

from rlpyt.algos.base import RlAlgorithm
from rlpyt.utils.quick_args import save_args
from rlpyt.algos.utils import discount_returns, valids_mean


class A2C(RlAlgorithm):

    def __init__(
            self, 
            discount=0.99,
            learning_rate=0.001,
            value_loss_coeff=0.5,
            entropy_loss_coeff=0.01,
            OptimCls=torch.optim.Adam,
            optim_kwargs=None,
            clip_grad_norm=5.,
            ):
        if optim_kwargs is None:
            optim_kwargs = dict()
        save_args(locals())

    def initialize(self, agent):
        # Put agent on GPU before making optimizer.
        self.optimizer = self.OptimCls(agent.parameters(), lr=self.learning_rate,
            **self.optim_kwargs)
        self.agent = agent  # any initialize?

    def loss(self, agent_samples, env_samples):
        returns, advantages, valids = self.process_samples(agent_samples, env_samples)
        pi, values = self.agent.forward(agent_samples, env_samples)

        dist = self.agent.distribution
        pi_loglis = dist.log_likelihood(pi, agent_samples.actions)
        pi_loss = - valids_mean(pi_loglis * advantages, valids)

        value_errors = 0.5 * (values - returns) ** 2
        value_loss = self.value_loss_coeff * valids_mean(value_errors, valids)

        entropies = dist.entropy(pi)
        entropy_loss = - self.entropy_loss_coeff * valids_mean(entropies, valids)

        loss = pi_loss + value_loss + entropy_loss
        return loss

    def optimize_agent(self, samples):
        self.optimizer.zero_grad()
        loss = self.loss(*samples)
        loss.backward()
        if self.clip_grad_norm:
            torch.nn.utils.clip_grad_norm_(self.agent.parameters(), self.clip_grad_norm)
        self.optimizer.step()
        # return diagnostics?

    def process_samples(self, agent_samples, env_samples):
        returns = discount_returns(env_samples.rewards, env_samples.dones,
            agent_samples.bootstrap_values, self.discount)
        advantages = returns - agent_samples.values
        if self.mid_batch_reset:
            valids = np.ones_like(env_samples.dones)
        else:
            valids = 1 - np.minimum(np.cumsum(env_samples.dones, axis=0), 1)
        return returns, advantages, valids

