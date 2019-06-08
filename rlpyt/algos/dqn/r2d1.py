
import torch

from rlpyt.aglos.base import RlAlgorithm
from rlpyt.utils.quick_args import save__init__args
from rlpyt.utils.logging import logger
from rlput.utils.collections import namedarraytuple
from rlpyt.replays.sequence.frame.uniform import UniformSequenceFrameBuffer
from rlpyt.replays.sequence.frame.prioritized import PrioritizedSequenceFrameBuffer
from rlpyt.utils.tensor import select_at_indexes


SamplesToReplay = namedarraytuple("SamplesToReplay",
    ["observation", "action", "reward", "done", "prev_rnn_state"])


class R2D1(RlAlgorithm):
    """Recurrent-replay DQN with options for: Double-DQN, Dueling Architecture,
    n-step returns, prioritized_replay."""

    def __init__(
            self,
            discount=0.997,
            batch_T=80,
            batch_B=64,
            warmup_T=40,
            use_stored_rnn_state=True,
            min_steps_learn=int(5e4),
            delta_clip=1.,
            replay_size=int(1e6),
            training_intensity=1,
            target_update_interval=2500,
            n_step_return=5,
            learning_rate=1e-4,
            OptimCls=torch.optim.Adam,
            optim_kwargs=None,
            grad_norm_clip=10.,
            eps_init=1,
            eps_final=0.01,
            eps_final_min=None,
            eps_steps=int(1e6),
            eps_eval=0.001,
            dueling_dqn=True,
            prioritized_replay=True,
            pri_alpha=0.6,
            pri_beta_init=0.9,
            pri_beta_final=0.9,
            pri_beta_steps=int(50e6),
            default_priority=None,
            ):
        if optim_kwargs is None:
            optim_kwargs = dict(eps=1e-3)  # Assumes Adam.
        if default_priority is None:
            default_priority = delta_clip
        save__init__args(locals())
        self.update_counter = 0

    def initialize(self, agent, n_itr, batch_spec, mid_batch_reset, examples):
        self.agent = agent
        if (self.eps_final_min is not None and
                self.eps_final_min != self.eps_final):  # vector-valued epsilon
            self.eps_init = self.eps_init * torch.ones(batch_spec.B)
            self.eps_final = torch.logspace(
                torch.log10(torch.tensor(self.eps_final_min)),
                torch.log10(torch.tensor(self.eps_final)),
                batch_spec.B)
        agent.set_epsilon_greedy(self.eps_init)
        agent.give_eval_epsilon_greedy(self.eps_eval)
        self.n_itr = n_itr
        self.optimizer = self.OptimCls(agent.parameters(),
            lr=self.learning_rate, **self.optim_kwargs)
        if self.initial_optim_state_dict is not None:
            self.optimizer.load_state_dict(self.initial_optim_state_dict)

        sample_bs = batch_spec.size
        train_bs = self.batch_T * self.batch_B
        assert (self.training_intensity * sample_bs) % train_bs == 0
        self.updates_per_optimize = int((self.training_intensity * sample_bs) //
            train_bs)
        logger.log(f"From sampler batch size {sample_bs}, training "
            f"batch size {train_bs}, and training intensity "
            f"{self.training_intensity}, computed {self.updates_per_optimize} "
            f"updates per iteration.")

        self.eps_itr = max(1, self.eps_steps // sample_bs)
        self.min_itr_learn = self.min_steps_learn // sample_bs
        if self.prioritized_replay:
            self.pri_beta_itr = max(1, self.pri_beta_steps // sample_bs)

        example_to_replay = SamplesToReplay(
            observation=examples["observation"],
            action=examples["action"],
            reward=examples["reward"],
            done=examples["done"],
            prev_rnn_state=examples["prev_rnn_state"],
        )
        replay_kwargs = dict(
            example=example_to_replay,
            size=self.replay_size,
            B=batch_spec.B,
            discount=self.discount,
            n_step_return=self.n_step_return,
        )
        if self.prioritize_replay:
            replay_kwargs.update(dict(
                alpha=self.pri_alpha,
                beta=self.pri_beta_init,
                default_priority=self.default_priority,
            ))
            ReplayCls = PrioritizedSequenceFrameBuffer
        else:
            ReplayCls = UniformSequenceFrameBuffer
        self.replay_buffer = ReplayCls(**replay_kwargs)

    def optimize_agent(self, samples, itr):
        samples_to_replay = SamplesToReplay(
            observation=samples.env.observation,
            action=samples.agent.action,
            reward=samples.env.reward,
            done=samples.env.done,
            prev_rnn_state=samples.agent.agent_info.prev_rnn_state,
        )
        self.replay_buffer.append_samples(samples_to_replay)
        if itr < self.min_itr_learn:
            return OptData(), OptInfo()  # TODO fix for empty
        opt_info = OptInfo(*([] for _ in range(len(OptInfo._fields))))
        for _ in range(self.updates_per_optimize):
            self.update_counter += 1
            samples_from_replay = self.replay_buffer.sample(self.batch_size)
            self.optimizer.zero_grad()
            loss, td_abs_errors, priorities = self.loss(samples_from_replay)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.agent.model.parameters(), self.clip_grad_norm)
            self.optimizer.step()
            if self.prioritized_replay:
                self.replay_buffer.update_batch_priorities(priorities)
            opt_info.loss.append(loss.item())
            opt_info.gradNorm.append(grad_norm)
            opt_info.tdAbsErr.extend(td_abs_erros[::8].numpy())
            opt_info.priority.extend(priorities)
            if self.update_counter % self.target_update_interval == 0:
                self.agent.update_target()
        self.update_itr_hyperparams(itr)
        return OptData(), opt_info

    def loss(self, samples):
        """Samples have leading Time and Batch dimentions [T,B,...]."""
        agent_inputs = buffer_to(samples.agent_inputs, device=self.agent.device)
        next_agent_inputs = buffer_to(samples.next_agent_inputs,
            device=self.agent.device)  # Move to device once, re-use.
        if self.use_stored_rnn_state:
            init_rnn_state = samples.init_rnn_state
            target_init_rnn_state = samples.next_init_rnn_state
        else:
            init_rnn_state = None
            target_init_rnn_state = None
        train_inputs = agent_inputs[self.warmup_T:]
        next_train_inputs = next_agent_inputs[self.warmup_T:]
        if self.warmup_T > 0:
            warmup_inputs = agent_inputs[:self.warmup_T]
            next_warmup_inputs = next_agent_inputs[:self.warmup_T]
            with torch.no_grad():
                init_rnn_state = self.agent.warmup(*warmup_inputs,
                    init_rnn_state)
                target_init_rnn_state = self.agent.target_warmup(
                    *next_warmup_inputs, target_init_rnn_state)
        qs = self.agent(*train_inputs, init_rnn_state)
        q = select_at_indexes(samples.action, qs)
        with torch.no_grad():
            target_qs = self.agent.target_q(*next_train_inputs,
                target_init_rnn_state)
            if self.double_dqn:
                next_qs = self.agent(*next_train_inputs)
                next_a = torch.argmax(next_qs, dim=-1)
                target_q = select_at_indexes(next_a, target_qs)
            else:
                target_q = torch.max(target_qs, dim=-1)
        disc = self.discount ** self.n_step_return
        y = h(samples.return_ + (1 - samples.done_n) * disc * h_inv(target_q))
        # TODO:  implement h, implenet rest of loss...is it Huber?
        # TODO:  get priorties by sequence.



