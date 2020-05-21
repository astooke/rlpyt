
import torch
from collections import namedtuple

from rlpyt.algos.base import RlAlgorithm
from rlpyt.utils.quick_args import save__init__args
from rlpyt.utils.logging import logger
from rlpyt.replays.non_sequence.uniform import (UniformReplayBuffer,
    AsyncUniformReplayBuffer)
from rlpyt.replays.non_sequence.time_limit import (TlUniformReplayBuffer,
    AsyncTlUniformReplayBuffer)
from rlpyt.utils.collections import namedarraytuple
from rlpyt.utils.tensor import valid_mean
from rlpyt.algos.utils import valid_from_done

OptInfo = namedtuple("OptInfo",
    ["muLoss", "qLoss", "muGradNorm", "qGradNorm"])
SamplesToBuffer = namedarraytuple("SamplesToBuffer",
    ["observation", "action", "reward", "done", "timeout"])


class DDPG(RlAlgorithm):
    """Deep deterministic policy gradient algorithm, training from a replay
    buffer."""

    opt_info_fields = tuple(f for f in OptInfo._fields)  # copy

    def __init__(
            self,
            discount=0.99,
            batch_size=64,
            min_steps_learn=int(1e4),
            replay_size=int(1e6),
            replay_ratio=64,  # data_consumption / data_generation
            target_update_tau=0.01,
            target_update_interval=1,
            policy_update_interval=1,
            learning_rate=1e-4,
            q_learning_rate=1e-3,
            OptimCls=torch.optim.Adam,
            optim_kwargs=None,
            initial_optim_state_dict=None,
            clip_grad_norm=1e8,
            q_target_clip=1e6,
            n_step_return=1,
            updates_per_sync=1,  # For async mode only.
            bootstrap_timelimit=True,
            ReplayBufferCls=None,
            ):
        """Saves input arguments."""
        if optim_kwargs is None:
            optim_kwargs = dict()
        self._batch_size = batch_size
        del batch_size  # Property.
        save__init__args(locals())

    def initialize(self, agent, n_itr, batch_spec, mid_batch_reset, examples,
            world_size=1, rank=0):
        """Stores input arguments and initializes replay buffer and optimizer.
        Use in non-async runners.  Computes number of gradient updates per
        optimization iteration as `(replay_ratio * sampler-batch-size /
        training-batch_size)`."""
        self.agent = agent
        self.n_itr = n_itr
        self.mid_batch_reset = mid_batch_reset
        self.sampler_bs = sampler_bs = batch_spec.size
        self.updates_per_optimize = max(1, round(self.replay_ratio * sampler_bs /
            self.batch_size))
        logger.log(f"From sampler batch size {sampler_bs}, training "
            f"batch size {self.batch_size}, and replay ratio "
            f"{self.replay_ratio}, computed {self.updates_per_optimize} "
            f"updates per iteration.")
        self.min_itr_learn = int(self.min_steps_learn // sampler_bs)
        # Agent give min itr learn.?
        self.initialize_replay_buffer(examples, batch_spec)
        self.optim_initialize(rank)

    def async_initialize(self, agent, sampler_n_itr, batch_spec, mid_batch_reset,
            examples, world_size=1):
        """Used in async runner only; returns replay buffer allocated in shared
        memory, does not instantiate optimizer. """
        self.agent = agent
        self.n_itr = sampler_n_itr
        self.initialize_replay_buffer(examples, batch_spec, async_=True)
        self.mid_batch_reset = mid_batch_reset
        self.sampler_bs = sampler_bs = batch_spec.size
        self.updates_per_optimize = self.updates_per_sync
        self.min_itr_learn = int(self.min_steps_learn // sampler_bs)
        return self.replay_buffer

    def optim_initialize(self, rank=0):
        """Called in initilize or by async runner after forking sampler."""
        self.rank = rank
        self.mu_optimizer = self.OptimCls(self.agent.mu_parameters(),
            lr=self.learning_rate, **self.optim_kwargs)
        self.q_optimizer = self.OptimCls(self.agent.q_parameters(),
            lr=self.q_learning_rate, **self.optim_kwargs)
        if self.initial_optim_state_dict is not None:
            self.q_optimizer.load_state_dict(self.initial_optim_state_dict["q"])
            self.mu_optimizer.load_state_dict(self.initial_optim_state_dict["mu"])

    def initialize_replay_buffer(self, examples, batch_spec, async_=False):
        """
        Allocates replay buffer using examples and with the fields in `SamplesToBuffer`
        namedarraytuple.
        """
        example_to_buffer = SamplesToBuffer(
            observation=examples["observation"],
            action=examples["action"],
            reward=examples["reward"],
            done=examples["done"],
            timeout=getattr(examples["env_info"], "timeout", None)
        )
        replay_kwargs = dict(
            example=example_to_buffer,
            size=self.replay_size,
            B=batch_spec.B,
            n_step_return=self.n_step_return,
        )
        if not self.bootstrap_timelimit:
            ReplayCls = AsyncUniformReplayBuffer if async_ else UniformReplayBuffer
        else:
            ReplayCls = AsyncTlUniformReplayBuffer if async_ else TlUniformReplayBuffer
        if self.ReplayBufferCls is not None:
            ReplayCls = self.ReplayBufferCls
            logger.log(f"WARNING: ignoring internal selection logic and using"
                f" input replay buffer class: {ReplayCls} -- compatibility not"
                " guaranteed.")
        self.replay_buffer = ReplayCls(**replay_kwargs)

    def optimize_agent(self, itr, samples=None, sampler_itr=None):
        """
        Extracts the needed fields from input samples and stores them in the 
        replay buffer.  Then samples from the replay buffer to train the agent
        by gradient updates (with the number of updates determined by replay
        ratio, sampler batch size, and training batch size).
        """
        itr = itr if sampler_itr is None else sampler_itr  # Async uses sampler_itr.
        if samples is not None:
            samples_to_buffer = self.samples_to_buffer(samples)
            self.replay_buffer.append_samples(samples_to_buffer)
        opt_info = OptInfo(*([] for _ in range(len(OptInfo._fields))))
        if itr < self.min_itr_learn:
            return opt_info
        for _ in range(self.updates_per_optimize):
            samples_from_replay = self.replay_buffer.sample_batch(self.batch_size)
            if self.mid_batch_reset and not self.agent.recurrent:
                valid = torch.ones_like(samples_from_replay.done, dtype=torch.float)
            else:
                valid = valid_from_done(samples_from_replay.done)
            if self.bootstrap_timelimit:
                # To avoid non-use of bootstrap when environment is 'done' due to
                # time-limit, turn off training on these samples.
                valid *= (1 - samples_from_replay.timeout_n.float())
            self.q_optimizer.zero_grad()
            q_loss = self.q_loss(samples_from_replay, valid)
            q_loss.backward()
            q_grad_norm = torch.nn.utils.clip_grad_norm_(
                self.agent.q_parameters(), self.clip_grad_norm)
            self.q_optimizer.step()
            opt_info.qLoss.append(q_loss.item())
            opt_info.qGradNorm.append(q_grad_norm)
            self.update_counter += 1
            if self.update_counter % self.policy_update_interval == 0:
                self.mu_optimizer.zero_grad()
                mu_loss = self.mu_loss(samples_from_replay, valid)
                mu_loss.backward()
                mu_grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.agent.mu_parameters(), self.clip_grad_norm)
                self.mu_optimizer.step()
                opt_info.muLoss.append(mu_loss.item())
                opt_info.muGradNorm.append(mu_grad_norm)
            if self.update_counter % self.target_update_interval == 0:
                self.agent.update_target(self.target_update_tau)
        return opt_info

    def samples_to_buffer(self, samples):
        """Defines how to add data from sampler into the replay buffer. Called
        in optimize_agent() if samples are provided to that method."""
        return SamplesToBuffer(
            observation=samples.env.observation,
            action=samples.agent.action,
            reward=samples.env.reward,
            done=samples.env.done,
            timeout=getattr(samples.env.env_info, "timeout", None)
        )

    def mu_loss(self, samples, valid):
        """Computes the mu_loss as the Q-value at that action."""
        mu_losses = self.agent.q_at_mu(*samples.agent_inputs)
        mu_loss = valid_mean(mu_losses, valid)  # valid can be None.
        return -mu_loss

    def q_loss(self, samples, valid):
        """Constructs the n-step Q-learning loss using target Q.  Input
        samples have leading batch dimension [B,..] (but not time)."""
        q = self.agent.q(*samples.agent_inputs, samples.action)
        with torch.no_grad():
            target_q = self.agent.target_q_at_mu(*samples.target_inputs)
        disc = self.discount ** self.n_step_return
        y = samples.return_ + (1 - samples.done_n.float()) * disc * target_q
        y = torch.clamp(y, -self.q_target_clip, self.q_target_clip)
        q_losses = 0.5 * (y - q) ** 2
        q_loss = valid_mean(q_losses, valid)  # valid can be None.
        return q_loss

    def optim_state_dict(self):
        return dict(q=self.q_optimizer.state_dict(),
            mu=self.mu_optimizer.state_dict())

    def load_optim_state_dict(self, state_dict):
        self.q_optimizer.load_state_dict(state_dict["q"])
        self.mu_optimizer.load_state_dict(state_dict["mu"])
