
######################################################################
# Algorithm file.
######################################################################


import torch
# import torch.nn.functional as F
from collections import namedtuple, deque

from rlpyt.algos.pg.base import PolicyGradientAlgo, OptInfo
from rlpyt.agents.base import AgentInputs
from rlpyt.utils.tensor import valid_mean
from rlpyt.utils.quick_args import save__init__args
from rlpyt.utils.buffer import buffer_to, buffer_method
from rlpyt.utils.collections import namedarraytuple
from rlpyt.utils.misc import iterate_mb_idxs
from rlpyt.algos.utils import (discount_return,
    generalized_advantage_estimation, valid_from_done)
from rlpyt.models.utils import strip_ddp_state_dict

OptInfoCost = namedtuple("OptInfoCost", OptInfo._fields + ("costPenalty",
    "costLimit", "valueError", "cvalueError", "valueAbsError", "cvalueAbsError",
    "pid_i", "pid_p", "pid_d", "pid_o", "betaKL", "betaKlRaw", "betaKlR",
    "betaKlC", "betaGradRaw", "betaGrad"))

LossInputs = namedarraytuple("LossInputs",
    ["agent_inputs", "action", "return_", "advantage", "valid", "old_dist_info",
    "c_return", "c_advantage"])


class CppoPID(PolicyGradientAlgo):

    opt_info_fields = OptInfoCost._fields

    def __init__(
            self,
            discount=0.99,
            learning_rate=0.001,
            value_loss_coeff=1.,
            entropy_loss_coeff=0.,
            OptimCls=torch.optim.Adam,
            optim_kwargs=None,
            clip_grad_norm=1.,
            initial_optim_state_dict=None,
            gae_lambda=0.97,
            minibatches=1,
            epochs=8,
            ratio_clip=0.1,
            linear_lr_schedule=False,
            normalize_advantage=False,
            cost_discount=None,  # if None, defaults to discount.
            cost_gae_lambda=None,
            cost_value_loss_coeff=None,
            ep_cost_ema_alpha=0,  # 0 for hard update, 1 for no update.
            objective_penalized=True,  # False for reward-only learning
            learn_c_value=True,  # Also False for reward-only learning
            penalty_init=1.,
            cost_limit=25,
            cost_scale=1.,  # divides; applied to raw cost and cost_limit
            normalize_cost_advantage=False,
            pid_Kp=0,
            pid_Ki=1,
            pid_Kd=0,
            pid_d_delay=10,
            pid_delta_p_ema_alpha=0.95,  # 0 for hard update, 1 for no update
            pid_delta_d_ema_alpha=0.95,
            sum_norm=True,  # L = (J_r - lam * J_c) / (1 + lam); lam <= 0
            diff_norm=False,  # L = (1 - lam) * J_r - lam * J_c; 0 <= lam <= 1
            penalty_max=100,  # only used if sum_norm=diff_norm=False
            step_cost_limit_steps=None,  # Change the cost limit partway through
            step_cost_limit_value=None,  # New value.
            use_beta_kl=False,
            use_beta_grad=False,
            record_beta_kl=False,
            record_beta_grad=False,
            beta_max=10,
            beta_ema_alpha=0.9,
            beta_kl_epochs=1,
            reward_scale=1,  # multiplicative (unlike cost_scale)
            lagrange_quadratic_penalty=False,
            quadratic_penalty_coeff=1,
            ):
        assert learn_c_value or not objective_penalized
        assert (step_cost_limit_steps is None) == (step_cost_limit_value is None)
        assert not (sum_norm and diff_norm)
        assert not (use_beta_kl and use_beta_grad)
        cost_discount = discount if cost_discount is None else cost_discount
        cost_gae_lambda = (gae_lambda if cost_gae_lambda is None else
            cost_gae_lambda)
        cost_value_loss_coeff = (value_loss_coeff if cost_value_loss_coeff is
            None else cost_value_loss_coeff)
        if optim_kwargs is None:
            optim_kwargs = dict()
        save__init__args(locals())
        self.cost_limit /= self.cost_scale
        if step_cost_limit_value is not None:
            self.step_cost_limit_value /= self.cost_scale
        self._beta_kl = 1.
        self._beta_grad = 1.
        self.beta_min = 1. / self.beta_max

    def initialize(self, *args, **kwargs):
        super().initialize(*args, **kwargs)
        self._batch_size = self.batch_spec.size // self.minibatches  # For logging.
        if self.linear_lr_schedule:
            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer=self.optimizer,
                lr_lambda=lambda itr: (self.n_itr - itr) / self.n_itr)  # Step once per itr.
            self._ratio_clip = self.ratio_clip  # Save base value.
        if self.step_cost_limit_steps is None:
            self.step_cost_limit_itr = None
        else:
            self.step_cost_limit_itr = int(self.step_cost_limit_steps //
                (self.batch_spec.size * self.world_size))
            # print("\n\n step cost itr: ", self.step_cost_limit_itr, "\n\n")
        self._ep_cost_ema = self.cost_limit  # No derivative at start.
        self._ddp = self.agent._ddp
        self.pid_i = self.cost_penalty = self.penalty_init
        self.cost_ds = deque(maxlen=self.pid_d_delay)
        self.cost_ds.append(0)
        self._delta_p = 0
        self._cost_d = 0
        if self.use_beta_kl or self.record_beta_kl:
            self.beta_r_optimizer = self.OptimCls(
                self.agent.beta_r_model.parameters(),
                lr=self.learning_rate, **self.optim_kwargs)
            self.beta_c_optimizer = self.OptimCls(
                self.agent.beta_c_model.parameters(),
                lr=self.learning_rate, **self.optim_kwargs)

    def optimize_agent(self, itr, samples):
        recurrent = self.agent.recurrent
        agent_inputs = AgentInputs(  # Move inputs to device once, index there.
            observation=samples.env.observation,
            prev_action=samples.agent.prev_action,
            prev_reward=samples.env.prev_reward,
        )
        agent_inputs = buffer_to(agent_inputs, device=self.agent.device)
        # return_, advantage, valid = self.process_returns(samples)
        (return_, advantage, valid, c_return, c_advantage,
            ep_cost_avg) = self.process_returns(itr, samples)
        loss_inputs = LossInputs(  # So can slice all.
            agent_inputs=agent_inputs,
            action=samples.agent.action,
            return_=return_,
            advantage=advantage,
            valid=valid,
            old_dist_info=samples.agent.agent_info.dist_info,
            c_return=c_return,  # Can be None.
            c_advantage=c_advantage,
        )
        opt_info = OptInfoCost(*([] for _ in range(len(OptInfoCost._fields))))

        if (self.step_cost_limit_itr is not None and
                self.step_cost_limit_itr == itr):
            self.cost_limit = self.step_cost_limit_value
        opt_info.costLimit.append(self.cost_limit)

        # PID update here:
        delta = float(ep_cost_avg - self.cost_limit)  # ep_cost_avg: tensor
        self.pid_i = max(0., self.pid_i + delta * self.pid_Ki)
        if self.diff_norm:
            self.pid_i = max(0., min(1., self.pid_i))
        a_p = self.pid_delta_p_ema_alpha
        self._delta_p *= a_p
        self._delta_p += (1 - a_p) * delta
        a_d = self.pid_delta_d_ema_alpha
        self._cost_d *= a_d
        self._cost_d += (1 - a_d) * float(ep_cost_avg)
        pid_d = max(0., self._cost_d - self.cost_ds[0])
        pid_o = (self.pid_Kp * self._delta_p + self.pid_i +
            self.pid_Kd * pid_d)
        self.cost_penalty = max(0., pid_o)
        if self.diff_norm:
            self.cost_penalty = min(1., self.cost_penalty)
        if not (self.diff_norm or self.sum_norm):
            self.cost_penalty = min(self.cost_penalty, self.penalty_max)
        self.cost_ds.append(self._cost_d)
        opt_info.pid_i.append(self.pid_i)
        opt_info.pid_p.append(self._delta_p)
        opt_info.pid_d.append(pid_d)
        opt_info.pid_o.append(pid_o)

        opt_info.costPenalty.append(self.cost_penalty)

        if hasattr(self.agent, "update_obs_rms"):
            self.agent.update_obs_rms(agent_inputs.observation)
            if itr == 0:
                return opt_info  # Sacrifice the first batch to get obs stats.

        if recurrent:
            # Leave in [B,N,H] for slicing to minibatches.
            init_rnn_state = samples.agent.agent_info.prev_rnn_state[0]  # T=0.
        T, B = samples.env.reward.shape[:2]
        # If recurrent, use whole trajectories, only shuffle B; else shuffle all.
        batch_size = B if self.agent.recurrent else T * B
        mb_size = batch_size // self.minibatches

        if self.use_beta_kl or self.record_beta_kl:
            raw_beta_kl, beta_r_kl, beta_c_kl = self.compute_beta_kl(
                loss_inputs, init_rnn_state, batch_size, mb_size, T)
            beta_KL = min(self.beta_max, max(self.beta_min, raw_beta_kl))
            self._beta_kl *= self.beta_ema_alpha
            self._beta_kl += (1 - self.beta_ema_alpha) * beta_KL
            opt_info.betaKlRaw.append(raw_beta_kl)
            opt_info.betaKL.append(self._beta_kl)
            opt_info.betaKlR.append(beta_r_kl)
            opt_info.betaKlC.append(beta_c_kl)
            # print("raw_beta_kl: ", raw_beta_kl)
            # print("self._beta_kl: ", self._beta_kl, "\n\n")

        if self.use_beta_grad or self.record_beta_grad:
            raw_beta_grad = self.compute_beta_grad(loss_inputs, init_rnn_state)
            beta_grad = min(self.beta_max, max(self.beta_min, raw_beta_grad))
            self._beta_grad *= self.beta_ema_alpha
            self._beta_grad += (1 - self.beta_ema_alpha) * beta_grad
            opt_info.betaGradRaw.append(raw_beta_grad)
            opt_info.betaGrad.append(self._beta_grad)

        for _ in range(self.epochs):
            for idxs in iterate_mb_idxs(batch_size, mb_size, shuffle=True):
                T_idxs = slice(None) if recurrent else idxs % T
                B_idxs = idxs if recurrent else idxs // T
                self.optimizer.zero_grad()
                rnn_state = init_rnn_state[B_idxs] if recurrent else None
                # NOTE: if not recurrent, will lose leading T dim, should be OK.
                loss, entropy, perplexity, value_errors, abs_value_errors = self.loss(
                    *loss_inputs[T_idxs, B_idxs], rnn_state)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.agent.parameters(), self.clip_grad_norm)
                self.optimizer.step()

                opt_info.loss.append(loss.item())
                opt_info.gradNorm.append(grad_norm)
                opt_info.entropy.append(entropy.item())
                opt_info.perplexity.append(perplexity.item())
                opt_info.valueError.extend(value_errors[0][::10].numpy())
                opt_info.cvalueError.extend(value_errors[1][::10].numpy())
                opt_info.valueAbsError.extend(abs_value_errors[0][::10].numpy())
                opt_info.cvalueAbsError.extend(abs_value_errors[1][::10].numpy())

                self.update_counter += 1
        if self.linear_lr_schedule:
            self.lr_scheduler.step()
            self.ratio_clip = self._ratio_clip * (self.n_itr - itr) / self.n_itr

        return opt_info

    def loss(self, agent_inputs, action, return_, advantage, valid, old_dist_info,
            c_return, c_advantage, init_rnn_state=None):
        if init_rnn_state is not None:
            # [B,N,H] --> [N,B,H] (for cudnn).
            init_rnn_state = buffer_method(init_rnn_state, "transpose", 0, 1)
            init_rnn_state = buffer_method(init_rnn_state, "contiguous")
            dist_info, value, _rnn_state = self.agent(*agent_inputs, init_rnn_state)
        else:
            dist_info, value = self.agent(*agent_inputs)
        dist = self.agent.distribution

        ratio = dist.likelihood_ratio(action, old_dist_info=old_dist_info,
            new_dist_info=dist_info)
        surr_1 = ratio * advantage
        clipped_ratio = torch.clamp(ratio, 1. - self.ratio_clip,
            1. + self.ratio_clip)
        surr_2 = clipped_ratio * advantage
        surrogate = torch.min(surr_1, surr_2)
        pi_loss = - valid_mean(surrogate, valid)

        if self.reward_scale == 1.:
            value_error = value.value - return_
        else:
            value_error = value.value - (return_ / self.reward_scale)  # Undo the scaling
        value_se = 0.5 * value_error ** 2
        value_loss = self.value_loss_coeff * valid_mean(value_se, valid)
        # Hmm, but with reward scaling, now the value gradient will be relatively smaller
        # than the pi gradient, unless we also change the value_loss_coeff??  Eh, leave it.

        entropy = dist.mean_entropy(dist_info, valid)
        entropy_loss = - self.entropy_loss_coeff * entropy

        if self.objective_penalized:
            # This, or just add c_advantage into advantage?
            c_surr_1 = ratio * c_advantage
            c_surr_2 = clipped_ratio * c_advantage
            c_surrogate = torch.max(c_surr_1, c_surr_2)
            c_loss = self.cost_penalty * valid_mean(c_surrogate, valid)
            if self.use_beta_kl:
                c_loss *= self._beta_kl
            elif self.use_beta_grad:
                c_loss *= self._beta_grad
            if self.diff_norm:  # (1 - lam) * R + lam * C
                pi_loss *= (1 - self.cost_penalty)
                pi_loss += c_loss
            elif self.sum_norm:  # 1 / (1 + lam) * (R + lam * C)
                pi_loss += c_loss
                pi_loss /= (1 + self.cost_penalty)
            else:
                pi_loss += c_loss

            if self.lagrange_quadratic_penalty:
                quad_loss = (self.quadratic_penalty_coeff
                    * valid_mean(c_surrogate, valid)
                    * torch.max(torch.tensor(0.), self._ep_cost_ema - self.cost_limit))
                pi_loss += quad_loss

        loss = pi_loss + value_loss + entropy_loss

        if self.learn_c_value:  # Then separate cost value estimate.
            assert value.c_value is not None
            assert c_return is not None
            c_value_error = value.c_value - c_return
            c_value_se = 0.5 * c_value_error ** 2
            c_value_loss = self.cost_value_loss_coeff * valid_mean(
                c_value_se, valid)
            loss += c_value_loss

        value_errors = (value_error.detach(), c_value_error.detach())
        if valid is not None:
            valid_mask = valid > 0
            value_errors = tuple(v[valid_mask] for v in value_errors)
        else:
            value_errors = tuple(v.view(-1) for v in value_errors)
        abs_value_errors = tuple(abs(v) for v in value_errors)
        perplexity = dist.mean_perplexity(dist_info, valid)
        return loss, entropy, perplexity, value_errors, abs_value_errors

    def process_returns(self, itr, samples):
        reward, cost = samples.env.reward, samples.env.env_info.cost
        cost /= self.cost_scale
        done = samples.env.done
        value, c_value = samples.agent.agent_info.value  # A named 2-tuple.
        bv, c_bv = samples.agent.bootstrap_value  # A named 2-tuple.

        if self.reward_scale != 1:
            reward *= self.reward_scale
            value *= self.reward_scale  # Keep the value learning the same.
            bv *= self.reward_scale

        done = done.type(reward.dtype)  # rlpyt does this in discount_returns?

        if c_value is not None:  # Learning c_value, even if reward penalized.
            if self.cost_gae_lambda == 1:  # GAE reduces to empirical discount.
                c_return = discount_return(cost, done, c_bv,
                    self.cost_discount)
                c_advantage = c_return - c_value
            else:
                c_advantage, c_return = generalized_advantage_estimation(
                    cost, c_value, done, c_bv, self.cost_discount,
                    self.cost_gae_lambda)
        else:
            c_advantage = c_return = None

        if self.gae_lambda == 1:  # GAE reduces to empirical discounted.
            return_ = discount_return(reward, done, bv, self.discount)
            advantage = return_ - value
        else:
            advantage, return_ = generalized_advantage_estimation(
                reward, value, done, bv, self.discount, self.gae_lambda)

        if not self.mid_batch_reset or self.agent.recurrent:
            valid = valid_from_done(done)  # Recurrent: no reset during training.
            # "done" might stay True until env resets next batch.
            # Could probably do this formula directly on (1 - done) and use it
            # regardless of mid_batch_reset.
            ep_cost_mask = valid * (1 - torch.cat([valid[1:],
                torch.ones_like(valid[-1:])]))  # Find where valid turns OFF.
        else:
            valid = None  # OR: torch.ones_like(done)
            ep_cost_mask = done  # Everywhere a done, is episode final cost.
        ep_costs = samples.env.env_info.cum_cost[ep_cost_mask.type(torch.bool)]

        if self._ddp:
            world_size = torch.distributed.get_world_size()  # already have self.world_size
        if ep_costs.numel() > 0:  # Might not have any completed trajectories.
            ep_cost_avg = ep_costs.mean()
            ep_cost_avg /= self.cost_scale
            if self._ddp:
                eca = ep_cost_avg.to(self.agent.device)
                torch.distributed.all_reduce(eca)
                ep_cost_avg = eca.to("cpu")
                ep_cost_avg /= world_size
            a = self.ep_cost_ema_alpha
            self._ep_cost_ema *= a
            self._ep_cost_ema += (1 - a) * ep_cost_avg

        if self.normalize_advantage:
            if valid is not None:
                valid_mask = valid > 0
                adv_mean = advantage[valid_mask].mean()
                adv_std = advantage[valid_mask].std()
            else:
                adv_mean = advantage.mean()
                adv_std = advantage.std()
            if self._ddp:
                mean_std = torch.stack([adv_mean, adv_std])
                mean_std = mean_std.to(self.agent.device)
                torch.distributed.all_reduce(mean_std)
                mean_std = mean_std.to("cpu")
                mean_std /= world_size
                adv_mean, adv_std = mean_std[0], mean_std[1]
            advantage[:] = (advantage - adv_mean) / max(adv_std, 1e-6)

        # Pretty sure not supposed to normalized c_advantage.
        if self.normalize_cost_advantage:
            if valid is not None:
                valid_mask = valid > 0
                cadv_mean = c_advantage[valid_mask].mean()
                cadv_std = c_advantage[valid_mask].std()
            else:
                cadv_mean = c_advantage.mean()
                cadv_std = c_advantage.std()
            if self._ddp:
                mean_std = torch.stack([cadv_mean, cadv_std])
                mean_std = mean_std.to(self.agent.device)
                torch.distributed.all_reduce(mean_std)
                mean_std = mean_std.to("cpu")
                mean_std /= world_size
                cadv_mean, cadv_std = mean_std[0], mean_std[1]
            c_advantage[:] = (c_advantage - cadv_mean) / max(cadv_std, 1e-6)

        return (return_, advantage, valid, c_return, c_advantage,
            self._ep_cost_ema)

    def compute_beta_kl(self, loss_inputs, init_rnn_state,
            batch_size, mb_size, T):
        """Ratio of KL divergences from reward-only vs cost-only updates."""
        self.agent.beta_r_model.load_state_dict(strip_ddp_state_dict(
            self.agent.model.state_dict()))
        self.agent.beta_c_model.load_state_dict(strip_ddp_state_dict(
            self.agent.model.state_dict()))
        self.beta_r_optimizer.load_state_dict(self.optimizer.state_dict())
        self.beta_c_optimizer.load_state_dict(self.optimizer.state_dict())

        recurrent = self.agent.recurrent
        for _ in range(self.beta_kl_epochs):
            for idxs in iterate_mb_idxs(batch_size, mb_size,
                    shuffle=batch_size > mb_size):
                T_idxs = slice(None) if recurrent else idxs % T
                B_idxs = idxs if recurrent else idxs // T
                rnn_state = init_rnn_state[B_idxs] if recurrent else None
                self.beta_r_optimizer.zero_grad()
                self.beta_c_optimizer.zero_grad()

                beta_r_loss, beta_c_loss = self.beta_kl_losses(
                    *loss_inputs[T_idxs, B_idxs], rnn_state)

                beta_r_loss.backward()
                _ = torch.nn.utils.clip_grad_norm_(
                    self.agent.beta_r_model.parameters(), self.clip_grad_norm)
                self.beta_r_optimizer.step()

                beta_c_loss.backward()
                _ = torch.nn.utils.clip_grad_norm_(
                    self.agent.beta_c_model.parameters(), self.clip_grad_norm)
                self.beta_c_optimizer.step()

        if init_rnn_state is not None:
            # [B,N,H] --> [N,B,H] (for cudnn).
            init_rnn_state = buffer_method(init_rnn_state, "transpose", 0, 1)
            init_rnn_state = buffer_method(init_rnn_state, "contiguous")
            with torch.no_grad():
                r_dist_info, c_dist_info = self.agent.beta_dist_infos(
                    *loss_inputs.agent_inputs, init_rnn_state)
        else:
            with torch.no_grad():
                r_dist_info, c_dist_info = self.agent.beta_dist_infos(
                    *loss_inputs.agent_inputs, init_rnn_state)

        dist = self.agent.distribution
        beta_r_KL = dist.mean_kl(new_dist_info=r_dist_info,
            old_dist_info=loss_inputs.old_dist_info, valid=loss_inputs.valid)
        beta_c_KL = dist.mean_kl(new_dist_info=c_dist_info,
            old_dist_info=loss_inputs.old_dist_info, valid=loss_inputs.valid)

        if self._ddp:
            beta_KLs = torch.stack([beta_r_KL, beta_c_KL])
            beta_KLs = beta_KLs.to(self.agent.device)
            torch.distributed.all_reduce(beta_KLs)
            beta_KLs = beta_KLs.to("cpu")
            beta_KLs /= torch.distributed.get_world_size()
            beta_r_KL, beta_c_KL = beta_KLs[0], beta_KLs[1]

        raw_beta_KL = float(beta_r_KL / max(beta_c_KL, 1e-8))

        return raw_beta_KL, float(beta_r_KL), float(beta_c_KL)

    def beta_kl_losses(self, agent_inputs, action, return_, advantage, valid,
            old_dist_info, c_return, c_advantage, init_rnn_state=None):
        if init_rnn_state is not None:
            # [B,N,H] --> [N,B,H] (for cudnn).
            init_rnn_state = buffer_method(init_rnn_state, "transpose", 0, 1)
            init_rnn_state = buffer_method(init_rnn_state, "contiguous")
            r_dist_info, c_dist_info = self.agent.beta_dist_infos(
                *agent_inputs, init_rnn_state)
        else:
            r_dist_info, c_dist_info = self.agent.beta_dist_infos(
                *agent_inputs)
        dist = self.agent.distribution

        r_ratio = dist.likelihood_ratio(action, old_dist_info=old_dist_info,
            new_dist_info=r_dist_info)
        surr_1 = r_ratio * advantage
        r_clipped_ratio = torch.clamp(r_ratio, 1. - self.ratio_clip,
            1. + self.ratio_clip)
        surr_2 = r_clipped_ratio * advantage
        surrogate = torch.min(surr_1, surr_2)
        beta_r_loss = - valid_mean(surrogate, valid)

        c_ratio = dist.likelihood_ratio(action, old_dist_info=old_dist_info,
            new_dist_info=c_dist_info)
        c_surr_1 = c_ratio * c_advantage
        c_clipped_ratio = torch.clamp(c_ratio, 1. - self.ratio_clip,
            1. + self.ratio_clip)
        c_surr_2 = c_clipped_ratio * c_advantage
        c_surrogate = torch.max(c_surr_1, c_surr_2)
        beta_c_loss = valid_mean(c_surrogate, valid)

        return beta_r_loss, beta_c_loss

    def compute_beta_grad(self, loss_inputs, init_rnn_state):
        """Ratio of KL grad-norms from reward vs cost objectives."""
        # Assume minibatches=1.
        self.optimizer.zero_grad()

        r_loss, c_loss = self.beta_grad_losses(*loss_inputs, init_rnn_state)

        r_loss.backward(retain_graph=True)
        r_grad_norm = torch.nn.utils.clip_grad_norm_(
            self.agent.parameters(), self.clip_grad_norm)
        r_grad_norm = min(r_grad_norm, self.clip_grad_norm)
        self.optimizer.zero_grad()
        c_loss.backward()
        c_grad_norm = torch.nn.utils.clip_grad_norm_(
            self.agent.parameters(), self.clip_grad_norm)
        c_grad_norm = min(c_grad_norm, self.clip_grad_norm)
        self.optimizer.zero_grad()

        if self._ddp:
            raise NotImplementedError

        return float(r_grad_norm) / float(c_grad_norm)

    def beta_grad_losses(self, agent_inputs, action, return_, advantage, valid,
            old_dist_info, c_return, c_advantage, init_rnn_state=None):
        if init_rnn_state is not None:
            # [B,N,H] --> [N,B,H] (for cudnn).
            init_rnn_state = buffer_method(init_rnn_state, "transpose", 0, 1)
            init_rnn_state = buffer_method(init_rnn_state, "contiguous")
            dist_info, value, _rnn_state = self.agent(*agent_inputs, init_rnn_state)
        else:
            dist_info, value = self.agent(*agent_inputs)
        dist = self.agent.distribution

        ratio = dist.likelihood_ratio(action, old_dist_info=old_dist_info,
            new_dist_info=dist_info)
        surr_1 = ratio * advantage
        clipped_ratio = torch.clamp(ratio, 1. - self.ratio_clip,
            1. + self.ratio_clip)
        surr_2 = clipped_ratio * advantage
        surrogate = torch.min(surr_1, surr_2)
        r_loss = - valid_mean(surrogate, valid)

        c_surr_1 = ratio * c_advantage
        c_surr_2 = clipped_ratio * c_advantage
        c_surrogate = torch.max(c_surr_1, c_surr_2)
        c_loss = valid_mean(c_surrogate, valid)

        return r_loss, c_loss
