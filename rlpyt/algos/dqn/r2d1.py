
import torch
from collections import namedtuple

from rlpyt.algos.dqn.dqn import DQN, SamplesToBuffer
from rlpyt.agents.base import AgentInputs
from rlpyt.utils.quick_args import save__init__args
from rlpyt.utils.logging import logger
from rlpyt.utils.collections import namedarraytuple
from rlpyt.replays.sequence.frame import (UniformSequenceReplayFrameBuffer,
    PrioritizedSequenceReplayFrameBuffer, AsyncUniformSequenceReplayFrameBuffer,
    AsyncPrioritizedSequenceReplayFrameBuffer)
from rlpyt.utils.tensor import select_at_indexes, valid_mean
from rlpyt.algos.utils import valid_from_done, discount_return_n_step
from rlpyt.utils.buffer import buffer_to, buffer_method, torchify_buffer

OptInfo = namedtuple("OptInfo", ["loss", "gradNorm", "tdAbsErr", "priority"])
SamplesToBufferRnn = namedarraytuple("SamplesToBufferRnn",
    SamplesToBuffer._fields + ("prev_rnn_state",))
PrioritiesSamplesToBuffer = namedarraytuple("PrioritiesSamplesToBuffer",
    ["priorities", "samples"])


class R2D1(DQN):
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
            min_steps_learn=int(1e5),
            delta_clip=None,  # Typically use squared-error loss (Steven).
            replay_size=int(1e6),
            replay_ratio=1,
            target_update_interval=2500,  # (Steven says 2500 but maybe faster.)
            n_step_return=5,
            learning_rate=1e-4,
            OptimCls=torch.optim.Adam,
            optim_kwargs=None,
            initial_optim_state_dict=None,
            clip_grad_norm=80.,  # 80 (Steven).
            # eps_init=1,  # NOW IN AGENT.
            # eps_final=0.1,
            # eps_final_min=0.0005,
            # eps_eval=0.001,
            eps_steps=int(1e6),  # STILL IN ALGO; conver to itr, give to agent.
            double_dqn=True,
            prioritized_replay=True,
            pri_alpha=0.6,
            pri_beta_init=0.9,
            pri_beta_final=0.9,
            pri_beta_steps=int(50e6),
            pri_eta=0.9,
            default_priority=None,
            input_priorities=True,
            input_priority_shift=None,
            value_scale_eps=1e-3,  # 1e-3 (Steven).
            ReplayBufferCls=None,  # leave None to select by above options
            updates_per_sync=1,  # For async mode only.
            ):
        """Saves input arguments.

        Args:
            store_rnn_state_interval (int): store RNN state only once this many steps, to reduce memory usage; replay sequences will only begin at the steps with stored recurrent state.
        
        Note:
            Typically ran with ``store_rnn_state_interval`` equal to the sampler's ``batch_T``, 40.  Then every 40 steps
            can be the beginning of a replay sequence, and will be guaranteed to start with a valid RNN state.  Only reset
            the RNN state (and env) at the end of the sampler batch, so that the beginnings of episodes are trained on.
        """
        if optim_kwargs is None:
            optim_kwargs = dict(eps=1e-3)  # Assumes Adam.
        if default_priority is None:
            default_priority = delta_clip or 1.
        if input_priority_shift is None:
            input_priority_shift = warmup_T // store_rnn_state_interval
        save__init__args(locals())
        self._batch_size = (self.batch_T + self.warmup_T) * self.batch_B

    def initialize_replay_buffer(self, examples, batch_spec, async_=False):
        """Similar to DQN but uses replay buffers which return sequences, and
        stores the agent's recurrent state."""
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
            # batch_T fixed for prioritized, (relax if rnn_state_interval=1 or 0).
            batch_T=self.batch_T + self.warmup_T,
        )
        if self.prioritized_replay:
            replay_kwargs.update(dict(
                alpha=self.pri_alpha,
                beta=self.pri_beta_init,
                default_priority=self.default_priority,
                input_priorities=self.input_priorities,  # True/False.
                input_priority_shift=self.input_priority_shift,
            ))
            ReplayCls = (AsyncPrioritizedSequenceReplayFrameBuffer if async_
                else PrioritizedSequenceReplayFrameBuffer)
        else:
            ReplayCls = (AsyncUniformSequenceReplayFrameBuffer if async_
                else UniformSequenceReplayFrameBuffer)
        if self.ReplayBufferCls is not None:
            ReplayCls = self.ReplayBufferCls
            logger.log(f"WARNING: ignoring internal selection logic and using"
                f" input replay buffer class: {ReplayCls} -- compatibility not"
                " guaranteed.")
        self.replay_buffer = ReplayCls(**replay_kwargs)
        return self.replay_buffer

    def optimize_agent(self, itr, samples=None, sampler_itr=None):
        """
        Similar to DQN, except allows to compute the priorities of new samples
        as they enter the replay buffer (input priorities), instead of only once they are
        used in training (important because the replay-ratio is quite low, about 1,
        so must avoid un-informative samples).
        """

        # TODO: estimate priorities for samples entering the replay buffer.
        # Steven says: workers did this approximately by using the online
        # network only for td-errors (not the target network).
        # This could be tough since add samples before the priorities are ready
        # (next batch), and in async case workers must do it.
        itr = itr if sampler_itr is None else sampler_itr  # Async uses sampler_itr
        if samples is not None:
            samples_to_buffer = self.samples_to_buffer(samples)
            self.replay_buffer.append_samples(samples_to_buffer)
        opt_info = OptInfo(*([] for _ in range(len(OptInfo._fields))))
        if itr < self.min_itr_learn:
            return opt_info
        for _ in range(self.updates_per_optimize):
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
            self.update_counter += 1
            if self.update_counter % self.target_update_interval == 0:
                self.agent.update_target()
        self.update_itr_hyperparams(itr)
        return opt_info

    def samples_to_buffer(self, samples):
        samples_to_buffer = super().samples_to_buffer(samples)
        if self.store_rnn_state_interval > 0:
            samples_to_buffer = SamplesToBufferRnn(*samples_to_buffer,
                prev_rnn_state=samples.agent.agent_info.prev_rnn_state)
        if self.input_priorities:
            priorities = self.compute_input_priorities(samples)
            samples_to_buffer = PrioritiesSamplesToBuffer(
                priorities=priorities, samples=samples_to_buffer)
        return samples_to_buffer

    def compute_input_priorities(self, samples):
        """Used when putting new samples into the replay buffer.  Computes
        n-step TD-errors using recorded Q-values from online network and
        value scaling.  Weights the max and the mean TD-error over each sequence
        to make a single priority value for that sequence.  

        Note:
            Although the original R2D2 implementation used the entire
            80-step sequence to compute the input priorities, we ran R2D1 with 40
            time-step sample batches, and so computed the priority for each
            80-step training sequence based on one of the two 40-step halves.
            Algorithm argument ``input_priority_shift`` determines which 40-step
            half is used as the priority for the 80-step sequence.  (Since this 
            method might get executed by alternating memory copiers in async mode,
            don't carry internal state here, do all computation with only the samples
            available in input.  Could probably reduce to one memory copier and keep
            state there, if needed.)
        """

        # """Just for first input into replay buffer.
        # Simple 1-step return TD-errors using recorded Q-values from online
        # network and value scaling, with the T dimension reduced away (same
        # priority applied to all samples in this batch; whereever the rnn state
        # is kept--hopefully the first step--this priority will apply there).
        # The samples duration T might be less than the training segment, so
        # this is an approximation of an approximation, but hopefully will
        # capture the right behavior.
        # UPDATE 20190826: Trying using n-step returns.  For now using samples
        # with full n-step return available...later could also use partial
        # returns for samples at end of batch.  35/40 ain't bad tho.
        # Might not carry/use internal state here, because might get executed
        # by alternating memory copiers in async mode; do all with only the
        # samples avialable from input."""
        samples = torchify_buffer(samples)
        q = samples.agent.agent_info.q
        action = samples.agent.action
        q_max = torch.max(q, dim=-1).values
        q_at_a = select_at_indexes(action, q)
        return_n, done_n = discount_return_n_step(
            reward=samples.env.reward,
            done=samples.env.done,
            n_step=self.n_step_return,
            discount=self.discount,
            do_truncated=False,  # Only samples with full n-step return.
        )
        # y = self.value_scale(
        #     samples.env.reward[:-1] +
        #     (self.discount * (1 - samples.env.done[:-1].float()) *  # probably done.float()
        #         self.inv_value_scale(q_max[1:]))
        # )
        nm1 = max(1, self.n_step_return - 1)  # At least 1 bc don't have next Q.
        y = self.value_scale(return_n +
            (1 - done_n.float()) * self.inv_value_scale(q_max[nm1:]))
        delta = abs(q_at_a[:-nm1] - y)
        # NOTE: by default, with R2D1, use squared-error loss, delta_clip=None.
        if self.delta_clip is not None:  # Huber loss.
            delta = torch.clamp(delta, 0, self.delta_clip)
        valid = valid_from_done(samples.env.done[:-nm1])
        max_d = torch.max(delta * valid, dim=0).values
        mean_d = valid_mean(delta, valid, dim=0)  # Still high if less valid.
        priorities = self.pri_eta * max_d + (1 - self.pri_eta) * mean_d  # [B]
        return priorities.numpy()

    def loss(self, samples):
        """Samples have leading Time and Batch dimentions [T,B,..]. Move all
        samples to device first, and then slice for sub-sequences.  Use same
        init_rnn_state for agent and target; start both at same t.  Warmup the
        RNN state first on the warmup subsequence, then train on the remaining
        subsequence.

        Returns loss (usually use MSE, not Huber), TD-error absolute values,
        and new sequence-wise priorities, based on weighted sum of max and mean
        TD-error over the sequence."""
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
        if self.store_rnn_state_interval == 0:
            init_rnn_state = None
        else:
            # [B,N,H]-->[N,B,H] cudnn.
            init_rnn_state = buffer_method(samples.init_rnn_state, "transpose", 0, 1)
            init_rnn_state = buffer_method(init_rnn_state, "contiguous")
        if wT > 0:  # Do warmup.
            with torch.no_grad():
                _, target_rnn_state = self.agent.target(*warmup_inputs, init_rnn_state)
                _, init_rnn_state = self.agent(*warmup_inputs, init_rnn_state)
            # Recommend aligning sampling batch_T and store_rnn_interval with
            # warmup_T (and no mid_batch_reset), so that end of trajectory
            # during warmup leads to new trajectory beginning at start of
            # training segment of replay.
            warmup_invalid_mask = valid_from_done(samples.done[:wT])[-1] == 0  # [B]
            init_rnn_state[:, warmup_invalid_mask] = 0  # [N,B,H] (cudnn)
            target_rnn_state[:, warmup_invalid_mask] = 0
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
        # NOTE: by default, with R2D1, use squared-error loss, delta_clip=None.
        if self.delta_clip is not None:  # Huber loss.
            b = self.delta_clip * (abs_delta - self.delta_clip / 2)
            losses = torch.where(abs_delta <= self.delta_clip, losses, b)
        if self.prioritized_replay:
            losses *= samples.is_weights.unsqueeze(0)  # weights: [B] --> [1,B]
        valid = valid_from_done(samples.done[wT:])  # 0 after first done.
        loss = valid_mean(losses, valid)
        td_abs_errors = abs_delta.detach()
        if self.delta_clip is not None:
            td_abs_errors = torch.clamp(td_abs_errors, 0, self.delta_clip)  # [T,B]
        valid_td_abs_errors = td_abs_errors * valid
        max_d = torch.max(valid_td_abs_errors, dim=0).values
        mean_d = valid_mean(td_abs_errors, valid, dim=0)  # Still high if less valid.
        priorities = self.pri_eta * max_d + (1 - self.pri_eta) * mean_d  # [B]

        return loss, valid_td_abs_errors, priorities

    def value_scale(self, x):
        """Value scaling function to handle raw rewards across games (not clipped)."""
        return (torch.sign(x) * (torch.sqrt(abs(x) + 1) - 1) +
            self.value_scale_eps * x)

    def inv_value_scale(self, z):
        """Invert the value scaling."""
        return torch.sign(z) * (((torch.sqrt(1 + 4 * self.value_scale_eps *
            (abs(z) + 1 + self.value_scale_eps)) - 1) /
            (2 * self.value_scale_eps)) ** 2 - 1)
