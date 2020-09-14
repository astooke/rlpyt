
import torch
from collections import namedtuple
import copy
import math
import numpy as np

from rlpyt.algos.base import RlAlgorithm
from rlpyt.utils.quick_args import save__init__args
from rlpyt.utils.logging import logger
# from rlpyt.replays.non_sequence.frame import (UniformReplayFrameBuffer,
#     PrioritizedReplayFrameBuffer)
from rlpyt.replays.non_sequence.uniform import UniformReplayBuffer
from rlpyt.utils.collections import namedarraytuple
from rlpyt.utils.tensor import select_at_indexes, valid_mean
from rlpyt.algos.utils import valid_from_done

from rlpyt.ul.algos.utils.warmup_scheduler import GradualWarmupScheduler
from rlpyt.models.utils import update_state_dict
from rlpyt.ul.algos.utils.data_augs import random_shift
from rlpyt.utils.buffer import buffer_to
from rlpyt.ul.models.rl.ul_models import UlEncoderModel
from rlpyt.ul.models.ul.atc_models import ContrastModel
from rlpyt.ul.replays.rl_with_ul_replay import (DqnWithUlUniformReplayFrameBuffer,
    DqnWithUlPrioritizedReplayFrameBuffer)


IGNORE_INDEX = -100  # Mask contrast samples across episode boundary.

OptInfoRl = namedtuple("OptInfoRl", ["loss", "gradNorm", "tdAbsErr"])
OptInfoUl = namedtuple("OptInfoUl", ["ulLoss", "ulAccuracy", "ulGradNorm",
    "ulUpdates"])
OptInfo = namedtuple("OptInfo", OptInfoUl._fields + OptInfoRl._fields)

SamplesToBuffer = namedarraytuple("SamplesToBuffer",
    ["observation", "action", "reward", "done"])


class DqnUl(RlAlgorithm):
    """
    DQN algorithm trainig from a replay buffer, with options for double-dqn, n-step
    returns, and prioritized replay.
    """

    opt_info_fields = tuple(f for f in OptInfo._fields)  # copy

    def __init__(
            self,
            discount=0.99,
            batch_size=256,
            min_steps_rl=int(1e5),
            delta_clip=1.,
            replay_size=int(1e6),
            replay_ratio=8,  # data_consumption / data_generation.
            target_update_tau=1,
            target_update_interval=1000,
            n_step_return=1,
            learning_rate=1.5e-4,
            OptimCls=torch.optim.Adam,
            optim_kwargs=None,
            initial_optim_state_dict=None,
            clip_grad_norm=40.,
            # eps_init=1,  # NOW IN AGENT.
            # eps_final=0.01,
            # eps_final_min=None,  # set < eps_final to use vector-valued eps.
            # eps_eval=0.001,
            eps_steps=int(1e6),  # STILL IN ALGO (to convert to itr).
            double_dqn=False,
            prioritized_replay=False,
            pri_alpha=0.6,
            pri_beta_init=0.4,
            pri_beta_final=1.,
            pri_beta_steps=int(50e6),
            default_priority=None,
            ReplayBufferCls=None,  # Leave None to select by above options.
            updates_per_sync=1,  # For async mode only.
            use_frame_buffer=True,
            min_steps_ul=int(5e4),
            max_steps_ul=None,
            ul_learning_rate=1e-3,
            ul_update_schedule=None,
            ul_lr_schedule=None,
            ul_lr_warmup=0,
            ul_delta_T=3,
            ul_batch_B=32,
            ul_batch_T=16,
            ul_random_shift_prob=0.1,
            ul_random_shift_pad=4,
            ul_target_update_interval=1,
            ul_target_update_tau=0.01,
            ul_latent_size=256,
            ul_anchor_hidden_sizes=512,
            ul_clip_grad_norm=10.,
            ul_optim_kwargs=None,
            # ul_pri_alpha=0.,  # No prioritization for now
            # ul_pri_beta=1.,
            # ul_pri_n_step_return=1,
            UlEncoderCls=UlEncoderModel,
            UlContrastCls=ContrastModel,
            ):
        """Saves input arguments.  

        ``delta_clip`` selects the Huber loss; if ``None``, uses MSE.

        ``replay_ratio`` determines the ratio of data-consumption
        to data-generation.  For example, original DQN sampled 4 environment steps between
        each training update with batch-size 32, for a replay ratio of 8.

        """ 
        if optim_kwargs is None:
            optim_kwargs = dict(eps=0.01 / batch_size)
        if ul_optim_kwargs is None:
            ul_optim_kwargs = dict()
        if default_priority is None:
            default_priority = delta_clip
        self._batch_size = batch_size
        del batch_size  # Property.
        save__init__args(locals())
        self.update_counter = 0

    def initialize(self, agent, n_itr, batch_spec, mid_batch_reset, examples,
            world_size=1, rank=0):
        """Stores input arguments and initializes replay buffer and optimizer.
        Use in non-async runners.  Computes number of gradient updates per
        optimization iteration as `(replay_ratio * sampler-batch-size /
        training-batch_size)`."""
        self.agent = agent
        self.n_itr = n_itr
        self.sampler_bs = sampler_bs = batch_spec.size
        self.mid_batch_reset = mid_batch_reset
        self.updates_per_optimize = max(1, round(self.replay_ratio * sampler_bs /
            self.batch_size))
        logger.log(f"From sampler batch size {batch_spec.size}, training "
            f"batch size {self.batch_size}, and replay ratio "
            f"{self.replay_ratio}, computed {self.updates_per_optimize} "
            f"updates per iteration.")
        self.min_itr_rl = int(self.min_steps_rl // sampler_bs)
        self.min_itr_ul = int(self.min_steps_ul // sampler_bs)
        self.max_itr_ul = (self.n_itr + 1 if self.max_steps_ul is None else
            self.max_steps_ul // sampler_bs)
        if self.min_itr_rl == self.min_itr_ul:
            self.min_itr_rl += 1  # wait until next?
        eps_itr_max = max(1, int(self.eps_steps // sampler_bs))
        agent.set_epsilon_itr_min_max(self.min_itr_rl, eps_itr_max)
        self.initialize_replay_buffer(examples, batch_spec)
        
        self.ul_encoder = self.UlEncoderCls(
            conv=self.agent.model.conv,
            latent_size=self.ul_latent_size,
            conv_out_size=self.agent.model.conv_out_size,
        )
        self.ul_target_encoder = copy.deepcopy(self.ul_encoder)
        self.ul_contrast = self.UlContrastCls(
            latent_size=self.ul_latent_size,
            anchor_hidden_sizes=self.ul_anchor_hidden_sizes,
        )
        self.ul_encoder.to(self.agent.device)
        self.ul_target_encoder.to(self.agent.device)
        self.ul_contrast.to(self.agent.device)

        self.optim_initialize(rank)

    def async_initialize(self, agent, sampler_n_itr, batch_spec, mid_batch_reset,
            examples, world_size=1):
        """Used in async runner only; returns replay buffer allocated in shared
        memory, does not instantiate optimizer. """
        # self.agent = agent
        # self.n_itr = sampler_n_itr
        # self.initialize_replay_buffer(examples, batch_spec, async_=True)
        # self.mid_batch_reset = mid_batch_reset
        # self.sampler_bs = sampler_bs = batch_spec.size
        # self.updates_per_optimize = self.updates_per_sync
        # self.min_itr_rl = int(self.min_steps_rl // sampler_bs)
        # eps_itr_max = max(1, int(self.eps_steps // sampler_bs))
        # # Before any forking so all sub processes have epsilon schedule:
        # agent.set_epsilon_itr_min_max(self.min_itr_rl, eps_itr_max)
        # return self.replay_buffer
        raise NotImplementedError

    def optim_initialize(self, rank=0):
        """Called in initilize or by async runner after forking sampler."""
        self.rank = rank
        self.optimizer = self.OptimCls(self.agent.parameters(),
            lr=self.learning_rate, **self.optim_kwargs)
        if self.initial_optim_state_dict is not None:
            self.optimizer.load_state_dict(self.initial_optim_state_dict)
        if self.prioritized_replay:
            self.pri_beta_itr = max(1, self.pri_beta_steps // self.sampler_bs)

        self.ul_optimizer = self.OptimCls(
            self.ul_parameters(),
            lr=self.ul_learning_rate, **self.ul_optim_kwargs)    
        
        self.total_ul_updates = sum([self.compute_ul_update_schedule(itr)
            for itr in range(self.n_itr)])
        logger.log(f"Total number of UL updates to do: {self.total_ul_updates}.")
        self.ul_update_counter = 0
        self.ul_lr_scheduler = None
        if self.total_ul_updates > 0:
            if self.ul_lr_schedule == "linear":
                self.ul_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                    optimizer=self.ul_optimizer,
                    lr_lambda=lambda upd: (self.total_ul_updates - upd) / self.total_ul_updates,
                )
            elif self.ul_lr_schedule == "cosine":
                self.ul_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer=self.ul_optimizer,
                    T_max=self.total_ul_updates - self.ul_lr_warmup,
                )
            elif self.ul_lr_schedule is not None:
                raise NotImplementedError

            if self.ul_lr_warmup > 0:
                self.ul_lr_scheduler = GradualWarmupScheduler(
                    self.ul_optimizer,
                    multiplier=1,
                    total_epoch=self.ul_lr_warmup,  # actually n_updates
                    after_scheduler=self.ul_lr_scheduler,
                )

            if self.ul_lr_scheduler is not None:
                self.ul_optimizer.zero_grad()
                self.ul_optimizer.step()

            self.c_e_loss = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

    def initialize_replay_buffer(self, examples, batch_spec, async_=False):
        """
        Allocates replay buffer using examples and with the fields in `SamplesToBuffer`
        namedarraytuple.  Uses frame-wise buffers, so that only unique frames are stored,
        using less memory (usual observations are 4 most recent frames, with only newest
        frame distince from previous observation).
        """
        if async_:
            raise NotImplementedError
        example_to_buffer = self.examples_to_buffer(examples)
        replay_kwargs = dict(
            example=example_to_buffer,
            size=self.replay_size,
            B=batch_spec.B,
            discount=self.discount,
            n_step_return=self.n_step_return,
        )
        if self.prioritized_replay:
            replay_kwargs.update(dict(
                alpha=self.pri_alpha,
                beta=self.pri_beta_init,
                default_priority=self.default_priority,
            ))
            ReplayCls = DqnWithUlPrioritizedReplayFrameBuffer
        else:
            ReplayCls = DqnWithUlUniformReplayFrameBuffer
        if not self.use_frame_buffer:
            logger.log("Overriding, using non-frame uniform replay buffer")
            ReplayCls = UniformReplayBuffer
        self.replay_buffer = ReplayCls(
            ul_replay_T=self.ul_delta_T + self.ul_batch_T,
            **replay_kwargs
        )

    def optimize_agent(self, itr, samples):
        """
        Extracts the needed fields from input samples and stores them in the 
        replay buffer.  Then samples from the replay buffer to train the agent
        by gradient updates (with the number of updates determined by replay
        ratio, sampler batch size, and training batch size).  If using prioritized
        replay, updates the priorities for sampled training batches.
        """
        samples_to_buffer = self.samples_to_buffer(samples)
        self.replay_buffer.append_samples(samples_to_buffer)
        opt_info = OptInfo(*([] for _ in range(len(OptInfo._fields))))
        if itr >= self.min_itr_rl:
            opt_info_rl = self.rl_optimize(itr)
            opt_info = opt_info._replace(**opt_info_rl._asdict())
        if itr >= self.min_itr_ul:
            opt_info_ul = self.ul_optimize(itr)
            opt_info = opt_info._replace(**opt_info_ul._asdict())
        else:
            opt_info.ulUpdates.append(0)
        return opt_info

    def rl_optimize(self, itr):
        opt_info_rl = OptInfoRl(*([] for _ in range(len(OptInfoRl._fields))))
        for _ in range(self.updates_per_optimize):
            samples_from_replay = self.replay_buffer.sample_batch(self.batch_size)
            self.optimizer.zero_grad()
            loss, td_abs_errors = self.loss(samples_from_replay)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.agent.parameters(), self.clip_grad_norm)
            self.optimizer.step()
            if self.prioritized_replay:
                self.replay_buffer.update_batch_priorities(td_abs_errors)
            opt_info_rl.loss.append(loss.item())
            opt_info_rl.gradNorm.append(grad_norm.item())
            opt_info_rl.tdAbsErr.extend(td_abs_errors[::8].numpy())  # Downsample.
            self.update_counter += 1
            if self.update_counter % self.target_update_interval == 0:
                self.agent.update_target(self.target_update_tau)
        self.update_itr_hyperparams(itr)
        return opt_info_rl

    def ul_optimize(self, itr):
        opt_info_ul = OptInfoUl(*([] for _ in range(len(OptInfoUl._fields))))
        n_ul_updates = self.compute_ul_update_schedule(itr)
        for i in range(n_ul_updates):
            self.ul_update_counter += 1
            if self.ul_lr_scheduler is not None:
                self.ul_lr_scheduler.step(self.ul_update_counter)
            ul_loss, ul_accuracy, grad_norm = self.ul_optimize_one_step()
            opt_info_ul.ulLoss.append(ul_loss.item())
            opt_info_ul.ulAccuracy.append(ul_accuracy.item())
            opt_info_ul.ulGradNorm.append(grad_norm.item())
            if self.ul_update_counter % self.ul_target_update_interval == 0:
                update_state_dict(self.ul_target_encoder, self.ul_encoder.state_dict(),
                    self.ul_target_update_tau)
        opt_info_ul.ulUpdates.append(self.ul_update_counter)
        return opt_info_ul

    def ul_optimize_one_step(self):
        self.ul_optimizer.zero_grad()
        samples = self.replay_buffer.ul_sample_batch(self.ul_batch_B)

        anchor = samples.observation[:-self.ul_delta_T]
        positive = samples.observation[self.ul_delta_T:]
        t, b, c, h, w = anchor.shape
        anchor = anchor.reshape(t * b, c, h, w)
        positive = positive.reshape(t * b, c, h, w)

        if self.ul_random_shift_prob > 0.:
            anchor = random_shift(
                imgs=anchor,
                pad=self.ul_random_shift_pad,
                prob=self.ul_random_shift_prob,
            )
            positive = random_shift(
                imgs=positive,
                pad=self.ul_random_shift_pad,
                prob=self.ul_random_shift_prob,
            )

        anchor, positive = buffer_to((anchor, positive),
            device=self.agent.device)

        with torch.no_grad():
            c_positive, _pos_conv = self.ul_target_encoder(positive)
        c_anchor, _anc_conv = self.ul_encoder(anchor)
        logits = self.ul_contrast(c_anchor, c_positive)  # anchor mlp in here.

        labels = torch.arange(c_anchor.shape[0],
            dtype=torch.long, device=self.agent.device)
        valid = valid_from_done(samples.done).type(torch.bool)  # use all
        valid = valid[self.ul_delta_T:].reshape(-1)  # at positions of positive
        labels[~valid] = IGNORE_INDEX
        
        ul_loss = self.c_e_loss(logits, labels)
        ul_loss.backward()
        if self.ul_clip_grad_norm is None:
            grad_norm = 0.
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.ul_parameters(), self.ul_clip_grad_norm)
        self.ul_optimizer.step()

        correct = torch.argmax(logits.detach(), dim=1) == labels
        accuracy = torch.mean(correct[valid].float())

        return ul_loss, accuracy, grad_norm

    def samples_to_buffer(self, samples):
        """Defines how to add data from sampler into the replay buffer. Called
        in optimize_agent() if samples are provided to that method.  In 
        asynchronous mode, will be called in the memory_copier process."""
        return SamplesToBuffer(
            observation=samples.env.observation,
            action=samples.agent.action,
            reward=samples.env.reward,
            done=samples.env.done,
        )

    def examples_to_buffer(self, examples):
        return SamplesToBuffer(
            observation=examples["observation"],
            action=examples["action"],
            reward=examples["reward"],
            done=examples["done"],
        )

    def loss(self, samples):
        """
        Computes the Q-learning loss, based on: 0.5 * (Q - target_Q) ^ 2.
        Implements regular DQN or Double-DQN for computing target_Q values
        using the agent's target network.  Computes the Huber loss using 
        ``delta_clip``, or if ``None``, uses MSE.  When using prioritized
        replay, multiplies losses by importance sample weights.

        Input ``samples`` have leading batch dimension [B,..] (but not time).

        Calls the agent to compute forward pass on training inputs, and calls
        ``agent.target()`` to compute target values.

        Returns loss and TD-absolute-errors for use in prioritization.

        Warning: 
            If not using mid_batch_reset, the sampler will only reset environments
            between iterations, so some samples in the replay buffer will be
            invalid.  This case is not supported here currently.
        """
        qs = self.agent(*samples.agent_inputs)
        q = select_at_indexes(samples.action, qs)
        with torch.no_grad():
            target_qs = self.agent.target(*samples.target_inputs)
            if self.double_dqn:
                next_qs = self.agent(*samples.target_inputs)
                next_a = torch.argmax(next_qs, dim=-1)
                target_q = select_at_indexes(next_a, target_qs)
            else:
                target_q = torch.max(target_qs, dim=-1).values
        disc_target_q = (self.discount ** self.n_step_return) * target_q
        y = samples.return_ + (1 - samples.done_n.float()) * disc_target_q
        delta = y - q
        losses = 0.5 * delta ** 2
        abs_delta = abs(delta)
        if self.delta_clip is not None:  # Huber loss.
            b = self.delta_clip * (abs_delta - self.delta_clip / 2)
            losses = torch.where(abs_delta <= self.delta_clip, losses, b)
        if self.prioritized_replay:
            losses *= samples.is_weights
        td_abs_errors = abs_delta.detach()
        if self.delta_clip is not None:
            td_abs_errors = torch.clamp(td_abs_errors, 0, self.delta_clip)
        if not self.mid_batch_reset:
            # FIXME: I think this is wrong, because the first "done" sample
            # is valid, but here there is no [T] dim, so there's no way to
            # know if a "done" sample is the first "done" in the sequence.
            raise NotImplementedError
            # valid = valid_from_done(samples.done)
            # loss = valid_mean(losses, valid)
            # td_abs_errors *= valid
        else:
            loss = torch.mean(losses)

        return loss, td_abs_errors

    def update_itr_hyperparams(self, itr):
        # EPS NOW IN AGENT.
        # if itr <= self.eps_itr:  # Epsilon can be vector-valued.
        #     prog = min(1, max(0, itr - self.min_itr_rl) /
        #       (self.eps_itr - self.min_itr_rl))
        #     new_eps = prog * self.eps_final + (1 - prog) * self.eps_init
        #     self.agent.set_sample_epsilon_greedy(new_eps)
        if self.prioritized_replay and itr <= self.pri_beta_itr:
            prog = min(1, max(0, itr - self.min_itr_rl) /
                (self.pri_beta_itr - self.min_itr_rl))
            new_beta = (prog * self.pri_beta_final +
                (1 - prog) * self.pri_beta_init)
            self.replay_buffer.set_beta(new_beta)

    def ul_parameters(self):
        yield from self.ul_encoder.parameters()
        yield from self.ul_contrast.parameters()

    def ul_named_parameters(self):
        yield from self.ul_encoder.named_parameters()
        yield from self.ul_contrast.named_parameters()

    def compute_ul_update_schedule(self, itr):
        if itr < self.min_itr_ul or itr > self.max_itr_ul:
            return 0
        remaining = (self.max_itr_ul - itr) / (self.max_itr_ul - self.min_itr_ul)  # from 1 to 0
        if "constant" in self.ul_update_schedule:
            # Format: "constant_X", for X num updates per RL itr.
            n_ul_updates = int(self.ul_update_schedule.split("_")[1])
        elif "front" in self.ul_update_schedule:
            # Format: "front_X_Y", for X updates first itr, Y updates rest.
            entries = self.ul_update_schedule.split("_")
            if itr == self.min_itr_ul:
                n_ul_updates = int(entries[1])
            else:
                n_ul_updates = int(entries[2])
        elif "linear" in self.ul_update_schedule:
            first = int(self.ul_update_schedule.split("_")[1])
            n_ul_updates = int(np.round(first * remaining))
        elif "quadratic" in self.ul_update_schedule:
            first = int(self.ul_update_schedule.split("_")[1])
            n_ul_updates = int(np.round(first * remaining ** 2))
        elif "cosine" in self.ul_update_schedule:
            first = int(self.ul_update_schedule.split("_")[1])
            n_ul_updates = int(np.round(first * math.sin(math.pi / 2 * remaining)))
        elif "mod" in self.ul_update_schedule:
            first = int(self.ul_update_schedule.split("_")[1])
            n_ul_updates = 1 if itr % first == 0 else 0
        return n_ul_updates

    def optim_state_dict(self):
        return dict(
            model=self.optimizer.state_dict(),
            ul=self.ul_optimizer.state_dict(),
        )

    def load_optim_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict["model"])
        self.ul_optimizer.load_state_dict(state_dict["ul"])
