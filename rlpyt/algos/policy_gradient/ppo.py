
import torch

from rlpyt.algos.policy_gradient.base import (PolicyGradient,
    OptData, OptInfo)
from rlpyt.agents.base import AgentInputs
from rlpyt.utils.tensor import valid_mean
from rlpyt.utils.quick_args import save__init__args
from rlpyt.utils.buffer import buffer_to
from rlpyt.utils.collections import namedarraytuple

LossInputs = namedarraytuple("LossInputs", ["agent_inputs", "opt_data",
    "action", "old_dist_info"])


class PPO(PolicyGradient):

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
            minibatches=4,
            epochs=4,
            ):
        if optim_kwargs is None:
            optim_kwargs = dict()
        save__init__args(locals())

    def optimize_agent(self, samples, itr):
        agent_inputs = AgentInputs(
            observation=samples.env.observation,
            prev_action=samples.agent.prev_action,
            prev_reward=samples.env.prev_reward,
        )
        agent_inputs = buffer_to(agent_inputs, device=self.agent.device)  # Once.

        return_, advantage, valid = self.process_returns(samples)
        opt_data = OptData(return_=return_, advantage=advantage, valid=valid)
        loss_inputs = LossInputs(agent_inputs, opt_data,
            samples.agent.action, samples.agent.agent_info.dist_info)

        T, B = samples.env.reward.shape[:2]
        opt_info = OptInfo([], [], [], [])
        if self.agent.recurrent:  # shuffle along B, but not T dimension.
            mb_size = B // self.minibatches
            for _ in range(self.epochs):
                for idxs in iterate_mb_idxs(B, mb_size, shuffle=True):
                    self.optimizer.zero_grad()
                    loss, entropy, perplexity = self.loss(*loss_inputs[:, idxs])
                    loss.backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.agent.model.parameters(), self.clip_grad_norm)
                    self.optimizer.step()
                    losses.append(loss.item())





    def loss(self, agent_inputs, opt_data, action, old_dist_info):
        dist_info, value = self.agent(*agent_inputs)
        return_, advantage, valid = opt_data
        dist = self.agent.distribution
        ratio = dist.likelihood_ratio(action, old_dist_info=old_dist_info,
            new_dist_info=dist_info)

        surr_1 = ratio * advantage
        clipped_ratio = torch.clamp(ratio, 1. - self.clip_param,
            1. + self.clip_param)
        surr_2 = clipped_ratio * advantage
        surrogate = torch.min(surr_1, surr_2)
        pi_loss = - valid_mean(surrogate, valid)

        value_error = 0.5 * (value - return_) ** 2
        value_loss = self.value_loss_coeff * valid_mean(value_error, valid)

        entropy = dist.mean_entropy(dist_info, valid)
        entropy_loss = - self.entropy_loss_coeff * entropy

        loss = pi_loss + value_loss + entropy_loss

        perplexity = dist.mean_perplexity(dist_info, valid)
        return loss, entropy, perplexity





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


    def a2c_optimize_agent(self, train_samples, itr):
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
