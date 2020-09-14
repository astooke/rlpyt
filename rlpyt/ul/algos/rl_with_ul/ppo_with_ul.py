
import torch
from collections import namedtuple
import copy
import math
import numpy as np

from rlpyt.algos.pg.ppo import PPO
from rlpyt.algos.pg.base import OptInfo as OptInfoRl
from rlpyt.utils.quick_args import save__init__args
# from rlpyt.replays.sequence.uniform import UniformSequenceReplayBuffer
from rlpyt.ul.replays.rl_with_ul_replay import (RlWithUlUniformReplayBuffer,
    RlWithUlPrioritizedReplayBuffer)
from rlpyt.utils.collections import namedarraytuple
# from rlpyt.models.mlp import MlpModel
from rlpyt.utils.buffer import buffer_to
from rlpyt.algos.utils import valid_from_done
from rlpyt.models.utils import update_state_dict
from rlpyt.ul.algos.utils.data_augs import random_shift
from rlpyt.ul.models.rl.ul_models import UlEncoderModel
from rlpyt.ul.models.ul.atc_models import ContrastModel
from rlpyt.utils.logging import logger
from rlpyt.ul.algos.utils.warmup_scheduler import GradualWarmupScheduler


IGNORE_INDEX = -100  # Mask CPC samples across episode boundary.
OptInfoUl = namedtuple("OptInfoUl", ["ulLoss", "ulAccuracy", "ulGradNorm",
    "ulUpdates"])
OptInfo = namedtuple("OptInfo", OptInfoRl._fields + OptInfoUl._fields)
SamplesToBuffer = namedarraytuple("SamplesToBuffer",
    ["observation", "action", "reward", "done"])


class PpoUl(PPO):

    opt_info_fields = tuple(f for f in OptInfo._fields)

    def __init__(
            self,
            discount=0.99,
            learning_rate=0.001,
            value_loss_coeff=1.,
            entropy_loss_coeff=0.01,
            OptimCls=torch.optim.Adam,
            optim_kwargs=None,
            clip_grad_norm=10.,
            initial_optim_state_dict=None,
            gae_lambda=1,
            minibatches=4,
            epochs=4,
            ratio_clip=0.1,
            linear_lr_schedule=True,
            normalize_advantage=False,
            min_steps_rl=0,
            min_steps_ul=0,
            max_steps_ul=None,
            ul_learning_rate=0.001,
            ul_optim_kwargs=None,
            ul_replay_size=1e5,
            ul_update_schedule=None,
            ul_lr_schedule=None,
            ul_lr_warmup=0,
            ul_delta_T=1,
            ul_batch_B=512,
            ul_batch_T=1,
            ul_random_shift_prob=1.,
            ul_random_shift_pad=4,
            ul_target_update_interval=1,
            ul_target_update_tau=0.01,
            ul_latent_size=256,
            ul_anchor_hidden_sizes=512,
            ul_clip_grad_norm=10.,
            ul_pri_alpha=0.,
            ul_pri_beta=1.,
            ul_pri_n_step_return=1,
            UlEncoderCls=UlEncoderModel,
            UlContrastCls=ContrastModel,
        ):
        """Saves input settings."""
        if optim_kwargs is None:
            optim_kwargs = dict()
        if ul_optim_kwargs is None:
            ul_optim_kwargs = dict()
        save__init__args(locals())

    def initialize(self, agent, n_itr, batch_spec, mid_batch_reset, examples,
            world_size=1, rank=0):
        """
        Extends base ``initialize()`` to initialize learning rate schedule, if
        applicable.
        """
        self.agent = agent
        self.n_itr = n_itr
        self.batch_spec = batch_spec
        self.mid_batch_reset = mid_batch_reset

        self.optimizer = self.OptimCls(  # Keep same name as base PPO algo.
            self.agent.parameters(),  # Model knows whether to include conv.
            lr=self.learning_rate, **self.optim_kwargs)

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

        self.ul_optimizer = self.OptimCls(
            self.ul_parameters(),
            lr=self.ul_learning_rate, **self.ul_optim_kwargs)

        self._batch_size = self.batch_spec.size // self.minibatches  # For logging.
        self.initialize_replay_buffer(examples, batch_spec)

        if self.linear_lr_schedule:
            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer=self.optimizer,
                lr_lambda=lambda itr: (self.n_itr - itr) / self.n_itr)  # Step once per itr.
            self._ratio_clip = self.ratio_clip  # Save base value.

        self.min_itr_rl = self.min_steps_rl // (batch_spec.size * world_size)
        self.min_itr_ul = self.min_steps_ul // (batch_spec.size * world_size)
        self.min_itr_ul = max(
            self.min_itr_ul, 
            1 + (self.ul_batch_T + self.ul_delta_T) // batch_spec.T,
        )
        self.max_itr_ul = (self.n_itr + 1 if self.max_steps_ul is None else
            self.max_steps_ul // (batch_spec.size * world_size))
        if self.min_itr_rl == self.min_itr_ul:
            self.min_itr_rl += 1
        logger.log(f"Min itr RL: {self.min_itr_rl},  Min itr UL: {self.min_itr_ul}. "
            f"Max itr UL: {self.max_itr_ul} (n_itr: {self.n_itr}).")
        if self.min_itr_rl > 0:
            self.agent.set_act_uniform(True)
        
        self.ul_lr_scheduler = None
        self.ul_update_counter = 0
        self.total_ul_updates = sum([self.compute_ul_update_schedule(itr)
            for itr in range(self.n_itr)])
        logger.log(f"Total number of UL updates to do: {self.total_ul_updates}.")
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

        if self.initial_optim_state_dict is not None:
            self.optimizer.load_state_dict(self.initial_optim_state_dict)

    def initialize_replay_buffer(self, examples, batch_spec):
        example_to_buffer = self.examples_to_buffer(examples)
        replay_T = self.ul_delta_T + self.ul_batch_T
        if self.ul_pri_alpha == 0.:
            self.replay_buffer = RlWithUlUniformReplayBuffer(
                example=example_to_buffer,
                size=self.ul_replay_size,
                B=batch_spec.B,
                replay_T=replay_T,
            )
        else:
            self.replay_buffer = RlWithUlPrioritizedReplayBuffer(
                example=example_to_buffer,
                size=self.ul_replay_size,
                B=batch_spec.B,
                replay_T=replay_T,
                discount=self.discount,
                n_step_return=self.ul_pri_n_step_return,
                alpha=self.ul_pri_alpha,
                beta=self.ul_pri_beta,
            )

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

    def optimize_agent(self, itr, samples):
        samples_to_buffer = self.samples_to_buffer(samples)
        self.replay_buffer.append_samples(samples_to_buffer)
        opt_info = OptInfo(*([] for _ in range(len(OptInfo._fields))))
        if itr == self.min_itr_rl:
            self.agent.set_act_uniform(False)  # Start using the policy network.
        if itr >= self.min_itr_rl:  # Do RL update first, "on-policy"?
            if self.agent.store_latent == "conv":  # agent expects it for training
                rl_samples = samples._replace(
                    env=samples.env_.replace(
                        observation=samples.agent.agent_info.conv)
                )
            else:
                rl_samples = samples
            opt_info_rl = super().optimize_agent(itr, rl_samples)  # Regular PPO
            opt_info = opt_info._replace(**opt_info_rl._asdict())
        if itr >= self.min_itr_ul:
            opt_info_ul = self.ul_optimize(itr)
            opt_info = opt_info._replace(**opt_info_ul._asdict())
        else:
            opt_info.ulUpdates.append(0)
        return opt_info

    def ul_optimize(self, itr):
        opt_info_ul = OptInfoUl(*([] for _ in range(len(OptInfoUl._fields))))
        n_ul_updates = self.compute_ul_update_schedule(itr)
        for _ in range(n_ul_updates):
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
        elif "pulse" in self.ul_update_schedule:
            # Format: "pulse_X_Y", for Y updates every X steps.
            entries = self.ul_update_schedule.split("_")
            n_steps_skip = int(entries[1])
            n_itr_skip = n_steps_skip // self.batch_spec.size
            if (itr - self.min_itr_ul) % n_itr_skip == 0:
                n_ul_updates = int(entries[2])
            else:
                n_ul_updates = 0
        elif "linear" in self.ul_update_schedule:
            first = int(self.ul_update_schedule.split("_")[1])
            n_ul_updates = int(np.round(first * remaining))
        elif "quadratic" in self.ul_update_schedule:
            first = int(self.ul_update_schedule.split("_")[1])
            n_ul_updates = int(np.round(first * remaining ** 2))
        elif "cosine" in self.ul_update_schedule:
            first = int(self.ul_update_schedule.split("_")[1])
            n_ul_updates = int(np.round(first * math.sin(math.pi / 2 * remaining)))
        return n_ul_updates

    def ul_optimize_one_step(self):
        self.ul_optimizer.zero_grad()
        samples = self.replay_buffer.sample_batch(batch_B=self.ul_batch_B)

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

    def ul_parameters(self):
        yield from self.ul_encoder.parameters()
        yield from self.ul_contrast.parameters()

    def ul_named_parameters(self):
        yield from self.ul_encoder.named_parameters()
        yield from self.ul_contrast.named_parameters()

    # def rl_parameters(self):
    #     if self.rl_conv_grad:
    #         yield from self.agent.conv_parameters()
    #     yield from self.agent.policy_parameters()

    # def rl_named_parameters(self):
    #     if self.rl_conv_grad:
    #         yield from self.agent.conv_named_parameters()
    #     yield from self.agent.policy_named_parameters()
