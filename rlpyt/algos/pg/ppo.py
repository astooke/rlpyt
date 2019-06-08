
import torch

from rlpyt.algos.policy_gradient.base import (PolicyGradient,
    OptData, OptInfo)
from rlpyt.agents.base import AgentInputs
from rlpyt.agents.base_recurrent import AgentTrainInputs
from rlpyt.utils.tensor import valid_mean
from rlpyt.utils.quick_args import save__init__args
from rlpyt.utils.buffer import buffer_to
from rlpyt.utils.collections import namedarraytuple
from rlpyt.utils.misc import iterate_mb_idxs

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
            ratio_clip=0.1,
            ):
        if optim_kwargs is None:
            optim_kwargs = dict()
        save__init__args(locals())

    def optimize_agent(self, samples, itr):
        agent_inputs = AgentInputs(  # Move inputs to device once, index there.
            observation=samples.env.observation,
            prev_action=samples.agent.prev_action,
            prev_reward=samples.env.prev_reward,
        )
        if self.agent.recurrent:
            agent_inputs = AgentTrainInputs(*agent_inputs,
                init_rnn_state=samples.agent.agent_info.prev_rnn_state[0],  # T=0
            )
        agent_inputs = buffer_to(agent_inputs, device=self.agent.device)
        return_, advantage, valid = self.process_returns(samples)
        opt_data = OptData(return_=return_, advantage=advantage, valid=valid)
        loss_inputs = LossInputs(agent_inputs, opt_data,  # So can slice all.
            samples.agent.action, samples.agent.agent_info.dist_info)

        T, B = samples.env.reward.shape[:2]
        opt_info = OptInfo(*([] for _ in range(len(OptInfo._fields))))
        # If recurrent, use whole trajectories, only shuffle B; else shuffle all.
        batch_size = B if self.agent.recurrent else T * B
        mb_size = batch_size // self.minibatches
        for _ in range(self.epochs):
            for idxs in iterate_mb_idxs(batch_size, mb_size, shuffle=True):
                T_idxs = slice(None) if self.agent.recurrent else idxs // T
                B_idxs = idxs if self.agent.recurrent else idxs % T
                self.optimizer.zero_grad()
                # NOTE: if not recurrent, will lose leading T dim, should be OK.
                loss, entropy, perplexity = self.loss(
                    *loss_inputs[T_idxs, B_idxs])
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.agent.model.parameters(), self.clip_grad_norm)
                self.optimizer.step()

                opt_info.loss.append(loss.item())
                opt_info.gradNorm.append(grad_norm)
                opt_info.entropy.append(entropy.item())
                opt_info.perplexity.append(perplexity.item())

        return opt_data, opt_info

    def loss(self, agent_inputs, opt_data, action, old_dist_info):
        dist_info, value = self.agent(*agent_inputs)
        return_, advantage, valid = opt_data
        dist = self.agent.distribution

        ratio = dist.likelihood_ratio(action, old_dist_info=old_dist_info,
            new_dist_info=dist_info)
        surr_1 = ratio * advantage
        clipped_ratio = torch.clamp(ratio, 1. - self.ratio_clip,
            1. + self.ratio_clip)
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
