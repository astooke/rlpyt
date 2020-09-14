
import math
import numpy as np
import torch
from collections import namedtuple
import copy

from rlpyt.algos.base import RlAlgorithm
from rlpyt.utils.quick_args import save__init__args
from rlpyt.utils.logging import logger
from rlpyt.replays.non_sequence.uniform import UniformReplayBuffer
from rlpyt.replays.non_sequence.time_limit import TlUniformReplayBuffer
from rlpyt.ul.algos.utils.data_augs import random_shift
from rlpyt.distributions.gaussian import Gaussian
from rlpyt.distributions.gaussian import DistInfo as GaussianDistInfo
from rlpyt.utils.tensor import valid_mean
from rlpyt.algos.utils import valid_from_done
from rlpyt.models.utils import update_state_dict
from rlpyt.utils.buffer import buffer_to
from rlpyt.utils.collections import namedarraytuple
from rlpyt.ul.models.rl.ul_models import UlEncoderModel
from rlpyt.ul.models.ul.atc_models import ContrastModel
from rlpyt.ul.algos.utils.warmup_scheduler import GradualWarmupScheduler
from rlpyt.ul.replays.rl_with_ul_replay import RlWithUlPrioritizedReplayWrapper

IGNORE_INDEX = -100  # Mask CPC samples across episode boundary.
OptInfoUl = namedtuple("OptInfoUl", ["ulLoss", "ulAccuracy", "ulGradNorm",
    "ulUpdates"])
OptInfoRl = namedtuple("OptInfoRl",
    ["q1Loss", "q2Loss", "piLoss",
    "qGradNorm", "piGradNorm",
    "q1", "q2", "piMu", "piLogStd", "qMeanDiff", "alpha",
    ])
OptInfo = namedtuple("OptInfo", OptInfoUl._fields + OptInfoRl._fields)

SamplesToBuffer = namedarraytuple("SamplesToBuffer",
    ["observation", "action", "reward", "done"])
SamplesToBufferTl = namedarraytuple("SamplesToBufferTl",
    SamplesToBuffer._fields + ("timeout",))


def chain(*iterables):
    for itr in iterables:
        yield from itr


class SacWithUl(RlAlgorithm):

    opt_info_fields = tuple(f for f in OptInfo._fields)

    def __init__(
            self,
            discount=0.99,
            batch_size=512,
            # replay_ratio=512,  # data_consumption / data_generation
            # min_steps_learn=int(1e4),
            replay_size=int(1e5),
            target_update_tau=0.01,  # tau=1 for hard update.
            target_update_interval=2,
            actor_update_interval=2,
            OptimCls=torch.optim.Adam,
            initial_optim_state_dict=None,  # for all of them.
            action_prior="uniform",  # or "gaussian"
            reward_scale=1,
            target_entropy="auto",  # "auto", float, or None
            reparameterize=True,
            clip_grad_norm=1e6,
            n_step_return=1,
            bootstrap_timelimit=True,
            q_lr=1e-3,
            pi_lr=1e-3,
            alpha_lr=1e-4,
            q_beta=0.9,
            pi_beta=0.9,
            alpha_beta=0.5,
            alpha_init=0.1,
            encoder_update_tau=0.05,
            random_shift_prob=1.,
            random_shift_pad=4,  # how much to pad on each direction (like DrQ style)
            stop_rl_conv_grad=False, 
            min_steps_rl=int(1e4),
            min_steps_ul=int(1e4),
            max_steps_ul=None,
            ul_learning_rate=7e-4,
            ul_optim_kwargs=None,
            # ul_replay_size=1e5,
            ul_update_schedule=None,
            ul_lr_schedule=None,
            ul_lr_warmup=0,
            # ul_delta_T=1,  # Always 1
            # ul_batch_B=512,
            # ul_batch_T=1,  # Always 1
            ul_batch_size=512,
            ul_random_shift_prob=1.,
            ul_random_shift_pad=4,
            ul_target_update_interval=1,
            ul_target_update_tau=0.01,
            ul_latent_size=128,
            ul_anchor_hidden_sizes=512,
            ul_clip_grad_norm=10.,
            ul_pri_alpha=0.,
            ul_pri_beta=1.,
            ul_pri_n_step_return=1,
            ul_use_rl_samples=False,
            UlEncoderCls=UlEncoderModel,
            UlContrastCls=ContrastModel,
            ):
        # assert replay_ratio == batch_size  # Unless I want to change it.
        self._batch_size = batch_size
        del batch_size
        if ul_optim_kwargs is None:
            ul_optim_kwargs = dict()
        save__init__args(locals())
        self.replay_ratio = self.batch_size  # standard 1 update per itr.
        # assert ul_delta_T == n_step_return  # Just use the same replay buffer
        # assert ul_batch_T == 1  # This was fine in DMControl in RlFromUl

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
        self.updates_per_optimize = int(self.replay_ratio * sampler_bs /
            self.batch_size)
        logger.log(f"From sampler batch size {sampler_bs}, training "
            f"batch size {self.batch_size}, and replay ratio "
            f"{self.replay_ratio}, computed {self.updates_per_optimize} "
            f"updates per iteration.")
        self.min_itr_rl = self.min_steps_rl // sampler_bs
        self.min_itr_ul = self.min_steps_ul // sampler_bs
        self.max_itr_ul = (self.n_itr + 1 if self.max_steps_ul is None else
            self.max_steps_ul // sampler_bs)
        if self.min_itr_rl == self.min_itr_ul:
            self.min_itr_rl += 1  # Wait until the next
        agent.give_min_itr_learn(self.min_itr_rl)
        self.initialize_replay_buffer(examples, batch_spec)
        
        self.ul_encoder = self.UlEncoderCls(
            conv=self.agent.conv,
            latent_size=self.ul_latent_size,
            conv_out_size=self.agent.conv.output_size,
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

    def async_initialize(*args, **kwargs):
        raise NotImplementedError

    def optim_initialize(self, rank=0):
        """Called in initilize or by async runner after forking sampler."""
        self.rank = rank

        # Be very explicit about which parameters are optimized where.
        self.pi_optimizer = self.OptimCls(chain(
            self.agent.pi_fc1.parameters(),  # No conv.
            self.agent.pi_mlp.parameters(),
            ),
            lr=self.pi_lr, betas=(self.pi_beta, 0.999))
        self.q_optimizer = self.OptimCls(chain(
            () if self.stop_rl_conv_grad else self.agent.conv.parameters(),
            self.agent.q_fc1.parameters(),
            self.agent.q_mlps.parameters(),
            ),
            lr=self.q_lr, betas=(self.q_beta, 0.999),
        )

        self._log_alpha = torch.tensor(np.log(self.alpha_init),
            requires_grad=True)
        self._alpha = torch.exp(self._log_alpha.detach())
        self.alpha_optimizer = self.OptimCls((self._log_alpha,),
            lr=self.alpha_lr, betas=(self.alpha_beta, 0.999))

        if self.target_entropy == "auto":
            self.target_entropy = -np.prod(self.agent.env_spaces.action.shape)
        if self.initial_optim_state_dict is not None:
            self.load_optim_state_dict(self.initial_optim_state_dict)
        if self.action_prior == "gaussian":
            self.action_prior_distribution = Gaussian(
                dim=np.prod(self.agent.env_spaces.action.shape), std=1.)

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
        namedarraytuple.
        POSSIBLY CHANGE TO FRAME-BASED BUFFER (only if need memory, speed is fine).
        """
        if async_:
            raise NotImplementedError
        example_to_buffer = self.examples_to_buffer(examples)
        ReplayCls = TlUniformReplayBuffer if self.bootstrap_timelimit else UniformReplayBuffer
        replay_kwargs = dict(
            example=example_to_buffer,
            size=self.replay_size,
            B=batch_spec.B,
            n_step_return=self.n_step_return,
        )
        self.replay_buffer = ReplayCls(**replay_kwargs)
        if self.ul_pri_alpha > 0.:
            self.replay_buffer = RlWithUlPrioritizedReplayWrapper(
                replay_buffer=self.replay_buffer,
                n_step_return=self.ul_pri_n_step_return,
                alpha=self.ul_pri_alpha,
                beta=self.ul_pri_beta,
            )

    def optimize_agent(self, itr, samples):
        """
        Extracts the needed fields from input samples and stores them in the 
        replay buffer.  Then samples from the replay buffer to train the agent
        by gradient updates (with the number of updates determined by replay
        ratio, sampler batch size, and training batch size).

        DIFFERENCES FROM SAC:
          -Organizes optimizers a little differently, clarifies which parameters.
        """
        samples_to_buffer = self.samples_to_buffer(samples)
        self.replay_buffer.append_samples(samples_to_buffer)
        opt_info = OptInfo(*([] for _ in range(len(OptInfo._fields))))
        rl_samples = None
        if itr >= self.min_itr_rl:
            opt_info_rl, rl_samples = self.rl_optimize(itr)
            opt_info = opt_info._replace(**opt_info_rl._asdict())
        if itr >= self.min_itr_ul:
            opt_info_ul = self.ul_optimize(itr, rl_samples)
            opt_info = opt_info._replace(**opt_info_ul._asdict())
        else:
            opt_info.ulUpdates.append(0)
        return opt_info

    def rl_optimize(self, itr):
        opt_info_rl = OptInfoRl(*([] for _ in range(len(OptInfoRl._fields))))
        for _ in range(self.updates_per_optimize):
            # Sample from the replay buffer, center crop, and move to GPU.
            samples_from_replay = self.replay_buffer.sample_batch(self.batch_size)
            rl_samples = self.random_shift_rl_samples(samples_from_replay)
            rl_samples = self.samples_to_device(rl_samples)

            # Q-loss includes computing some values used in pi-loss.
            q1_loss, q2_loss, valid, conv_out, q1, q2 = self.q_loss(rl_samples)

            if self.update_counter % self.actor_update_interval == 0:
                pi_loss, alpha_loss, pi_mean, pi_log_std = self.pi_alpha_loss(
                    rl_samples, valid, conv_out)
                if alpha_loss is not None:
                    self.alpha_optimizer.zero_grad()
                    alpha_loss.backward()
                    self.alpha_optimizer.step()
                    self._alpha = torch.exp(self._log_alpha.detach())
                    opt_info_rl.alpha.append(self._alpha.item())

                self.pi_optimizer.zero_grad()
                pi_loss.backward()
                pi_grad_norm = torch.nn.utils.clip_grad_norm_(chain(
                    self.agent.pi_fc1.parameters(),
                    self.agent.pi_mlp.parameters(),
                    ),
                    self.clip_grad_norm)
                self.pi_optimizer.step()
                opt_info_rl.piLoss.append(pi_loss.item())
                opt_info_rl.piGradNorm.append(pi_grad_norm.item())
                opt_info_rl.piMu.extend(pi_mean[::10].numpy())
                opt_info_rl.piLogStd.extend(pi_log_std[::10].numpy())

            # Step Q's last because pi_loss.backward() uses them.
            self.q_optimizer.zero_grad()
            q_loss = q1_loss + q2_loss
            q_loss.backward()
            q_grad_norm = torch.nn.utils.clip_grad_norm_(chain(
                () if self.stop_rl_conv_grad else self.agent.conv.parameters(),
                self.agent.q_fc1.parameters(),
                self.agent.q_mlps.parameters(),
                ),
                self.clip_grad_norm)
            self.q_optimizer.step()
            opt_info_rl.q1Loss.append(q1_loss.item())
            opt_info_rl.q2Loss.append(q2_loss.item())
            opt_info_rl.qGradNorm.append(q_grad_norm.item())
            opt_info_rl.q1.extend(q1[::10].numpy())  # Downsample for stats.
            opt_info_rl.q2.extend(q2[::10].numpy())
            opt_info_rl.qMeanDiff.append(torch.mean(abs(q1 - q2)).item())

            self.update_counter += 1
            if self.update_counter % self.target_update_interval == 0:
                self.agent.update_targets(
                    q_tau=self.target_update_tau,
                    encoder_tau=self.encoder_update_tau,
                )

        return opt_info_rl, rl_samples

    def ul_optimize(self, itr, rl_samples=None):
        opt_info_ul = OptInfoUl(*([] for _ in range(len(OptInfoUl._fields))))
        n_ul_updates = self.compute_ul_update_schedule(itr)
        ul_bs = self.ul_batch_size
        n_rl_samples = 0 if rl_samples is None else len(rl_samples.agent_inputs.observation)
        for i in range(n_ul_updates):
            self.ul_update_counter += 1
            if self.ul_lr_scheduler is not None:
                self.ul_lr_scheduler.step(self.ul_update_counter)
            if n_rl_samples >= self.ul_batch_size * (i + 1):
                ul_samples = rl_samples[i * ul_bs:(i + 1) * ul_bs]
            else:
                ul_samples = None
            ul_loss, ul_accuracy, grad_norm = self.ul_optimize_one_step(ul_samples)
            opt_info_ul.ulLoss.append(ul_loss.item())
            opt_info_ul.ulAccuracy.append(ul_accuracy.item())
            opt_info_ul.ulGradNorm.append(grad_norm.item())
            if self.ul_update_counter % self.ul_target_update_interval == 0:
                update_state_dict(self.ul_target_encoder, self.ul_encoder.state_dict(),
                    self.ul_target_update_tau)
        opt_info_ul.ulUpdates.append(self.ul_update_counter)
        return opt_info_ul

    def ul_optimize_one_step(self, samples=None):
        self.ul_optimizer.zero_grad()
        if samples is None:
            if self.ul_pri_alpha > 0:
                samples = self.replay_buffer.sample_batch(self.ul_batch_size,
                    mode="UL")
            else:
                samples = self.replay_buffer.sample_batch(self.ul_batch_size)

            # This is why need ul_delta_T == n_step_return, usually == 1;
            anchor = samples.agent_inputs.observation
            positive = samples.target_inputs.observation

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

        else:
            # Assume samples were already augmented in the RL loss.
            anchor = samples.agent_inputs.observation
            positive = samples.target_inputs.observation

        with torch.no_grad():
            c_positive, _pos_conv = self.ul_target_encoder(positive)
        c_anchor, _anc_conv = self.ul_encoder(anchor)
        logits = self.ul_contrast(c_anchor, c_positive)  # anchor mlp in here.

        labels = torch.arange(c_anchor.shape[0],
            dtype=torch.long, device=self.agent.device)
        invalid = samples.done  # shape: [B], if done, following state invalid
        labels[invalid] = IGNORE_INDEX
        ul_loss = self.c_e_loss(logits, labels)
        ul_loss.backward()
        if self.ul_clip_grad_norm is None:
            grad_norm = 0.
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.ul_parameters(), self.ul_clip_grad_norm)
        self.ul_optimizer.step()

        correct = torch.argmax(logits.detach(), dim=1) == labels
        accuracy = torch.mean(correct[~invalid].float())

        return ul_loss, accuracy, grad_norm

    def samples_to_buffer(self, samples):
        """Defines how to add data from sampler into the replay buffer. Called
        in optimize_agent() if samples are provided to that method."""
        observation = samples.env.observation
        samples_to_buffer = SamplesToBuffer(
            observation=observation,
            action=samples.agent.action,
            reward=samples.env.reward,
            done=samples.env.done,
        )
        if self.bootstrap_timelimit:
            samples_to_buffer = SamplesToBufferTl(*samples_to_buffer,
                timeout=samples.env.env_info.timeout)
        return samples_to_buffer

    def examples_to_buffer(self, examples):
        observation = examples["observation"]
        example_to_buffer = SamplesToBuffer(
            observation=observation,
            action=examples["action"],
            reward=examples["reward"],
            done=examples["done"],
        )
        if self.bootstrap_timelimit:
            example_to_buffer = SamplesToBufferTl(*example_to_buffer,
                timeout=examples["env_info"].timeout)
        return example_to_buffer

    def samples_to_device(self, samples):
        """Only move the parts of samples which need to go to GPU."""
        agent_inputs, target_inputs, action = buffer_to(
            (samples.agent_inputs, samples.target_inputs, samples.action),
            device=self.agent.device,
        )
        device_samples = samples._replace(
            agent_inputs=agent_inputs,
            target_inputs=target_inputs,
            action=action,
        )
        return device_samples

    def random_shift_rl_samples(self, samples):
        if self.random_shift_prob == 0.:
            return samples
        obs = samples.agent_inputs.observation
        target_obs = samples.target_inputs.observation
        aug_obs = random_shift(
            imgs=obs,
            pad=self.random_shift_pad,
            prob=self.random_shift_prob,
        )
        aug_target_obs = random_shift(
            imgs=target_obs,
            pad=self.random_shift_pad,
            prob=self.random_shift_prob,
        )
        aug_samples = samples._replace(
            agent_inputs=samples.agent_inputs._replace(observation=aug_obs),
            target_inputs=samples.target_inputs._replace(observation=aug_target_obs),
        )
        return aug_samples

    def q_loss(self, samples):
        if self.mid_batch_reset and not self.agent.recurrent:
            valid = torch.ones_like(samples.done, dtype=torch.float)  # or None
        else:
            valid = valid_from_done(samples.done)
        if self.bootstrap_timelimit:
            # To avoid non-use of bootstrap when environment is 'done' due to
            # time-limit, turn off training on these samples.
            valid *= (1 - samples.timeout_n.float())

        # Run the convolution only once, return so pi_loss can use it.
        conv_out = self.agent.conv(samples.agent_inputs.observation)
        if self.stop_rl_conv_grad:
            conv_out = conv_out.detach()
        q_inputs = samples.agent_inputs._replace(observation=conv_out)

        # Q LOSS.
        q1, q2 = self.agent.q(*q_inputs, samples.action)
        with torch.no_grad():
            # Run the target convolution only once.
            target_conv_out = self.agent.target_conv(samples.target_inputs.observation)
            target_inputs = samples.target_inputs._replace(observation=target_conv_out)
            target_action, target_log_pi, _ = self.agent.pi(*target_inputs)
            target_q1, target_q2 = self.agent.target_q(*target_inputs, target_action)
            min_target_q = torch.min(target_q1, target_q2)
            target_value = min_target_q - self._alpha * target_log_pi
        disc = self.discount ** self.n_step_return
        y = (self.reward_scale * samples.return_ +
            (1 - samples.done_n.float()) * disc * target_value)
        q1_loss = 0.5 * valid_mean((y - q1) ** 2, valid)
        q2_loss = 0.5 * valid_mean((y - q2) ** 2, valid)

        return q1_loss, q2_loss, valid, conv_out, q1.detach(), q2.detach()

    def pi_alpha_loss(self, samples, valid, conv_out):
        # PI LOSS.
        # Uses detached conv out; avoid re-computing.
        conv_detach = conv_out.detach()
        agent_inputs = samples.agent_inputs._replace(observation=conv_detach)
        new_action, log_pi, (pi_mean, pi_log_std) = self.agent.pi(*agent_inputs)
        if not self.reparameterize:
            # new_action = new_action.detach()  # No grad.
            raise NotImplementedError
        # Re-use the detached latent.
        log_target1, log_target2 = self.agent.q(*agent_inputs, new_action)
        min_log_target = torch.min(log_target1, log_target2)
        prior_log_pi = self.get_action_prior(new_action.cpu())
        if self.reparameterize:
            pi_losses = self._alpha * log_pi - min_log_target - prior_log_pi
        else:
            raise NotImplementedError
        # if self.policy_output_regularization > 0:
        #     pi_losses += self.policy_output_regularization * torch.mean(
        #         0.5 * pi_mean ** 2 + 0.5 * pi_log_std ** 2, dim=-1)
        pi_loss = valid_mean(pi_losses, valid)

        # ALPHA LOSS.
        if self.target_entropy is not None:
            alpha_losses = - self._log_alpha * (log_pi.detach() + self.target_entropy)
            alpha_loss = valid_mean(alpha_losses, valid)
        else:
            alpha_loss = None

        return pi_loss, alpha_loss, pi_mean.detach(), pi_log_std.detach()

    def get_action_prior(self, action):
        if self.action_prior == "uniform":
            prior_log_pi = 0.0
        elif self.action_prior == "gaussian":
            prior_log_pi = self.action_prior_distribution.log_likelihood(
                action, GaussianDistInfo(mean=torch.zeros_like(action)))
        return prior_log_pi

    def optim_state_dict(self):
        return dict(
            pi=self.pi_optimizer.state_dict(),
            q=self.q_optimizer.state_dict(),
            alpha=self.alpha_optimizer.state_dict(),
            log_alpha_value=self._log_alpha.detach().item(),
            ul=self.ul_optimizer.state_dict(),
        )

    def load_optim_state_dict(self, state_dict):
        self.pi_optimizer.load_state_dict(state_dict["pi"])
        self.q_optimizer.load_state_dict(state_dict["q"])
        self.alpha_optimizer.load_state_dict(state_dict["alpha"])
        self.ul_optimizer.load_state_dict(state_dict["ul"])
        with torch.no_grad():
            self._log_alpha[:] = state_dict["log_alpha_value"]
            self._alpha = torch.exp(self._log_alpha.detach())

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
        return n_ul_updates
