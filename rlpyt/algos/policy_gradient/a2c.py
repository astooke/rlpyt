
import torch

from rlpyt.algos.policy_gradient.base import (PolicyGradient,
    OptData, OptInfo)
from rlpyt.agents.base import AgentInputs
from rlpyt.utils.tensor import valid_mean
from rlpyt.utils.quick_args import save__init__args


class A2C(PolicyGradient):

    def __init__(
            self,
            discount=0.99,
            learning_rate=0.001,
            value_loss_coeff=0.5,
            entropy_loss_coeff=0.01,
            OptimCls=torch.optim.Adam,
            optim_kwargs=None,
            clip_grad_norm=1.,
            initial_optim_state_dict=None,
            gae_lambda=1,
            ):
        if optim_kwargs is None:
            optim_kwargs = dict()
        save__init__args(locals())

    def optimize_agent(self, train_samples, itr):
        self.optimizer.zero_grad()
        loss, entropy, perplexity, opt_data = self.loss(train_samples)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.agent.model.parameters(), self.clip_grad_norm)
        self.optimizer.step()
        opt_info = OptInfo(
            loss=loss.item(),
            gradNorm=grad_norm,
            entropy=entropy.item(),
            perplexity=perplexity.item(),
        )
        return opt_data, opt_info

    def loss(self, samples):
        agent_inputs = AgentInputs(samples.env.observation,
            samples.agent.prev_action, samples.env.prev_reward)
        dist_info, value = self.agent(*agent_inputs)
        # TODO: try to compute everyone on device.
        return_, advantage, valid = self.process_returns(samples)

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
