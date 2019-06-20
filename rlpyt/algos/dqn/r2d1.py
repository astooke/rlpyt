
import torch
from collections import namedtuple

from rlpyt.algos.base import RlAlgorithm
from rlpyt.agents.base import AgentInputs
from rlpyt.utils.quick_args import save__init__args
from rlpyt.utils.logging import logger
from rlpyt.utils.collections import namedarraytuple
from rlpyt.replays.sequence.frame import UniformSequenceReplayFrameBuffer
from rlpyt.replays.sequence.frame import PrioritizedSequenceReplayFrameBuffer
from rlpyt.utils.tensor import select_at_indexes, valid_mean
from rlpyt.algos.utils import valid_from_done
from rlpyt.utils.buffer import buffer_to

OptInfo = namedtuple("OptInfo", ["loss", "gradNorm", "tdAbsErr", "priority"])
SamplesToBuffer = namedarraytuple("SamplesToBuffer",
    ["observation", "action", "reward", "done"])
SamplesToBufferRnn = namedarraytuple("SamplesToBufferRnn",
    SamplesToBuffer._fields + ("prev_rnn_state",))


class R2D1(RlAlgorithm):
    """Recurrent-replay DQN with options for: Double-DQN, Dueling Architecture,
    n-step returns, prioritized_replay."""

    opt_info_fields = tuple(f for f in OptInfo._fields)  # copy

    def __init__(
            self,
            discount=0.997,
            batch_T=80,
            batch_B=64,
            warmup_T=40,
            store_rnn_state_interval=40,  # 0 for none, 1 for all.
            min_steps_learn=int(5e4),
            delta_clip=1.,
            replay_size=int(1e6),
            training_ratio=1,
            target_update_interval=2500,  # Might shrink this?
            n_step_return=5,
            learning_rate=1e-4,
            OptimCls=torch.optim.Adam,
            optim_kwargs=None,
            initial_optim_state_dict=None,
            clip_grad_norm=10.,
            eps_init=1,
            eps_final=0.05,
            eps_final_min=0.0005,
            eps_steps=int(1e6),
            eps_eval=0.001,
            double_dqn=True,
            prioritized_replay=True,
            pri_alpha=0.6,
            pri_beta_init=0.9,
            pri_beta_final=0.9,
            pri_beta_steps=int(50e6),
            pri_eta=0.9,
            default_priority=None,
            value_scale_eps=1e-2,
            ):
        if optim_kwargs is None:
            optim_kwargs = dict(eps=1e-3)  # Assumes Adam.
        if default_priority is None:
            default_priority = delta_clip
        save__init__args(locals())
        self.update_counter = 0

    def initialize(self, agent, n_itr, batch_spec, mid_batch_reset, examples):
        if not agent.recurrent:
            raise TypeError("For nonrecurrent agents use dqn (or fake recurrent).")
        self.agent = agent
        if (self.eps_final_min is not None and
                self.eps_final_min != self.eps_final):  # vector-valued epsilon
            self.eps_init = self.eps_init * torch.ones(batch_spec.B)
            self.eps_final = torch.logspace(
                torch.log10(torch.tensor(self.eps_final_min)),
                torch.log10(torch.tensor(self.eps_final)),
                batch_spec.B)
        agent.set_sample_epsilon_greedy(self.eps_init)
        agent.give_eval_epsilon_greedy(self.eps_eval)
        self.n_itr = n_itr
        self.mid_batch_reset = mid_batch_reset
        self.optimizer = self.OptimCls(agent.parameters(),
            lr=self.learning_rate, **self.optim_kwargs)
        if self.initial_optim_state_dict is not None:
            self.optimizer.load_state_dict(self.initial_optim_state_dict)

        sample_bs = batch_spec.size
        train_bs = (self.batch_T + self.warmup_T) * self.batch_B
        # assert (self.training_ratio * sample_bs) % train_bs == 0
        self.updates_per_optimize = max(1,
            round(self.training_ratio * sample_bs / train_bs))
        logger.log(f"From sampler batch size {sample_bs}, training "
            f"batch size {train_bs}, and training ratio "
            f"{self.training_ratio}, rounded to {self.updates_per_optimize} "
            f"updates per iteration.")

        self.eps_itr = max(1, self.eps_steps // sample_bs)
        self.min_itr_learn = self.min_steps_learn // sample_bs
        if self.prioritized_replay:
            self.pri_beta_itr = max(1, self.pri_beta_steps // sample_bs)

        example_to_buffer = SamplesToBuffer(
            observation=examples["observation"],
            action=examples["action"],
            reward=examples["reward"],
            done=examples["done"],
        )
        if self.store_rnn_state_interval > 0:
            example_to_buffer = SamplesToBufferRnn(*example_to_buffer,
                prev_rnn_state=examples["agent_info"].prev_rnn_state,
            )
        replay_kwargs = dict(
            example=example_to_buffer,
            size=self.replay_size,
            B=batch_spec.B,
            discount=self.discount,
            n_step_return=self.n_step_return,
            rnn_state_interval=self.store_rnn_state_interval,
            batch_T=self.batch_T + self.warmup_T,  # Fixed for prioritized replay.
        )
        if self.prioritized_replay:
            replay_kwargs.update(dict(
                alpha=self.pri_alpha,
                beta=self.pri_beta_init,
                default_priority=self.default_priority,
            ))
            ReplayCls = PrioritizedSequenceReplayFrameBuffer
        else:
            ReplayCls = UniformSequenceReplayFrameBuffer
        self.replay_buffer = ReplayCls(**replay_kwargs)

    def optimize_agent(self, samples, itr):
        samples_to_buffer = SamplesToBuffer(
            observation=samples.env.observation,
            action=samples.agent.action,
            reward=samples.env.reward,
            done=samples.env.done,
        )
        if self.store_rnn_state_interval > 0:
            samples_to_buffer = SamplesToBufferRnn(*samples_to_buffer,
                prev_rnn_state=samples.agent.agent_info.prev_rnn_state)
        self.replay_buffer.append_samples(samples_to_buffer)
        opt_info = OptInfo(*([] for _ in range(len(OptInfo._fields))))
        if itr < self.min_itr_learn:
            return opt_info
        for _ in range(self.updates_per_optimize):
            self.update_counter += 1
            samples_from_replay = self.replay_buffer.sample_batch(self.batch_B)
            self.optimizer.zero_grad()
            loss, td_abs_errors, priorities = self.loss(samples_from_replay)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.agent.parameters(), self.clip_grad_norm)
            self.optimizer.step()
            if self.prioritized_replay:
                self.replay_buffer.update_batch_priorities(priorities)
            opt_info.loss.append(loss.item())
            opt_info.gradNorm.append(grad_norm)
            opt_info.tdAbsErr.extend(td_abs_errors[::8].numpy())
            opt_info.priority.extend(priorities)
            if self.update_counter % self.target_update_interval == 0:
                self.agent.update_target()
        self.update_itr_hyperparams(itr)
        return opt_info

    def loss(self, samples):
        """Samples have leading Time and Batch dimentions [T,B,..]. Move all
        samples to device first, and then slice for sub-sequences.  Use same
        init_rnn_state for agent and target; start both at same t."""
        all_observation, all_action, all_reward = buffer_to(
            (samples.all_observation, samples.all_action, samples.all_reward),
            device=self.agent.device)
        wT, bT, nsr = self.warmup_T, self.batch_T, self.n_step_return
        if wT > 0:
            warmup_slice = slice(None, wT)  # Same for agent and target.
            warmup_inputs = AgentInputs(
                observation=all_observation[warmup_slice],
                prev_action=all_action[warmup_slice],
                prev_reward=all_reward[warmup_slice],
            )
        agent_slice = slice(wT, wT + bT)
        agent_inputs = AgentInputs(
            observation=all_observation[agent_slice],
            prev_action=all_action[agent_slice],
            prev_reward=all_reward[agent_slice],
        )
        target_slice = slice(wT, None)  # Same start t as agent. (wT + bT + nsr)
        target_inputs = AgentInputs(
            observation=all_observation[target_slice],
            prev_action=all_action[target_slice],
            prev_reward=all_reward[target_slice],
        )
        action = samples.all_action[wT + 1:wT + 1 + bT]  # CPU.
        return_ = samples.return_[wT:wT + bT]
        done_n = samples.done_n[wT:wT + bT]
        init_rnn_state = (None if self.store_rnn_state_interval == 0 else
            buffer_to(samples.init_rnn_state, device=self.agent.device))
        if wT > 0:
            with torch.no_grad():
                _, init_rnn_state = self.agent(*warmup_inputs, init_rnn_state)
                _, target_rnn_state = self.agent.target(*warmup_inputs,
                    init_rnn_state)
        else:
            target_rnn_state = init_rnn_state

        qs, _ = self.agent(*agent_inputs, init_rnn_state)  # [T,B,A]
        q = select_at_indexes(action, qs)
        with torch.no_grad():
            target_qs, _ = self.agent.target(*target_inputs, target_rnn_state)
            if self.double_dqn:
                next_qs, _ = self.agent(*target_inputs, init_rnn_state)
                next_a = torch.argmax(next_qs, dim=-1)
                target_q = select_at_indexes(next_a, target_qs)
            else:
                target_q = torch.max(target_qs, dim=-1).values
            target_q = target_q[-bT:]  # Same length as q.

        disc = self.discount ** self.n_step_return
        y = self.value_scale(return_ + (1 - done_n.float()) * disc *
            self.inv_value_scale(target_q))  # [T,B]
        delta = y - q
        losses = 0.5 * delta ** 2
        abs_delta = abs(delta)
        if self.delta_clip is not None:  # Huber loss.
            b = self.delta_clip * (abs_delta - self.delta_clip / 2)
            losses = torch.where(abs_delta <= self.delta_clip, losses, b)
        if self.prioritized_replay:
            losses *= samples.is_weights.unsqueeze(0)  # weights: [B] --> [1,B]
        valid = valid_from_done(samples.done)  # 0 after first done.
        loss = valid_mean(losses, valid)
        td_abs_errors = torch.clamp(abs_delta.detach(), 0, self.delta_clip)  # [T,B]
        valid_td_abs_errors = td_abs_errors * valid
        max_d = torch.max(valid_td_abs_errors, dim=0).values
        mean_d = torch.mean(valid_td_abs_errors, dim=0)  # Lower if less valid.
        priorities = self.pri_eta * max_d + (1 - self.pri_eta) * mean_d  # [B]

        return loss, valid_td_abs_errors, priorities

    def value_scale(self, x):
        return (torch.sign(x) * (torch.sqrt(abs(x) + 1) - 1) +
            self.value_scale_eps * x)

    def inv_value_scale(self, z):
        return torch.sign(z) * (((torch.sqrt(1 + 4 * self.value_scale_eps *
            (abs(z) + 1 + self.value_scale_eps)) - 1) /
            (2 * self.value_scale_eps)) ** 2 - 1)

    def update_itr_hyperparams(self, itr):
        if itr <= self.eps_itr:  # Epsilon can be vector-valued.
            prog = min(1, itr / self.eps_itr)
            new_eps = prog * self.eps_final + (1 - prog) * self.eps_init
            self.agent.set_sample_epsilon_greedy(new_eps)
        if self.prioritized_replay and itr <= self.pri_beta_itr:
            prog = min(1, itr / self.pri_beta_itr)
            new_beta = (prog * self.pri_beta_final +
                (1 - prog) * self.pri_beta_init)
            self.replay_buffer.set_beta(new_beta)
