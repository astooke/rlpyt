
import torch
import torch.nn.functional as F
from collections import namedtuple

from rlpyt.ul.algos.ul_for_rl.base import BaseUlAlgorithm
from rlpyt.ul.models.ul.vae_models import VaeHeadModel, VaeDecoderModel
from rlpyt.utils.quick_args import save__init__args
from rlpyt.utils.logging import logger
from rlpyt.ul.replays.ul_for_rl_replay import UlForRlReplayBuffer
from rlpyt.utils.buffer import buffer_to
from rlpyt.algos.utils import valid_from_done
from rlpyt.utils.tensor import valid_mean
from rlpyt.ul.models.ul.encoders import EncoderModel
from rlpyt.distributions.categorical import Categorical


# from rlpyt.ul.algos.data_augs import random_shift
# from rlpyt.distributions.categorical import Categorical, DistInfo


IGNORE_INDEX = -100  # Mask action samples across episode boundary.
OptInfo = namedtuple("OptInfo", ["reconLoss", "klLoss",
    "activationLoss", "gradNorm", "convActivation"])
ValInfo = namedtuple("ValInfo", ["reconLoss", "klLoss",
    "convActivation"])


class VAE(BaseUlAlgorithm):
    """VAE to predict o_t+k from o_t."""

    opt_info_fields = tuple(f for f in OptInfo._fields)  # copy

    def __init__(
            self,
            batch_size,
            learning_rate,
            replay_filepath,
            delta_T=0,
            OptimCls=torch.optim.Adam,
            optim_kwargs=None,
            initial_state_dict=None,
            clip_grad_norm=1000.,
            EncoderCls=EncoderModel,
            encoder_kwargs=None,
            latent_size=128,
            ReplayCls=UlForRlReplayBuffer,
            activation_loss_coefficient=0.0,
            learning_rate_anneal=None,  # cosine
            learning_rate_warmup=0,  # number of updates
            VaeHeadCls=VaeHeadModel,
            hidden_sizes=None,  # But maybe use for forward prediction
            DecoderCls=VaeDecoderModel,
            decoder_kwargs=None,
            kl_coeff=1.,
            onehot_action=True,
            validation_split=0.0,
            n_validation_batches=0,
            ):
        optim_kwargs = dict() if optim_kwargs is None else optim_kwargs
        encoder_kwargs = dict() if encoder_kwargs is None else encoder_kwargs
        decoder_kwargs = dict() if decoder_kwargs is None else decoder_kwargs
        save__init__args(locals())
        self.c_e_loss = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
        assert learning_rate_anneal in [None, "cosine"]
        self._replay_T = delta_T + 1

    def initialize(self, n_updates, cuda_idx=None):
        self.device = torch.device("cpu") if cuda_idx is None else torch.device(
            "cuda", index=cuda_idx)

        examples = self.load_replay()
        self.encoder = self.EncoderCls(
            image_shape=examples.observation.shape,
            latent_size=self.latent_size,  # UNUSED
            **self.encoder_kwargs
        )
        if self.onehot_action:
            act_dim = self.replay_buffer.samples.action.max() + 1  # discrete only
            self.distribution = Categorical(act_dim)
        else:
            act_shape = self.replay_buffer.samples.action.shape[2:]
            assert len(act_shape) == 1
            act_dim = act_shape[0]
        self.vae_head = self.VaeHeadCls(
            latent_size=self.latent_size,
            action_size=act_dim * self.delta_T,
            hidden_sizes=self.hidden_sizes,
        )
        self.decoder = self.DecoderCls(
            latent_size=self.latent_size,
            **self.decoder_kwargs
        )
        self.encoder.to(self.device)
        self.vae_head.to(self.device)
        self.decoder.to(self.device)

        self.optim_initialize(n_updates)

        if self.initial_state_dict is not None:
            self.load_state_dict(self.initial_state_dict)

    def optimize(self, itr):
        opt_info = OptInfo(*([] for _ in range(len(OptInfo._fields))))
        samples = self.replay_buffer.sample_batch(self.batch_size)
        if self.lr_scheduler is not None:
            self.lr_scheduler.step(itr)  # Do every itr instead of every epoch
        self.optimizer.zero_grad()
        recon_loss, kl_loss, conv_output = self.vae_loss(samples)
        act_loss = self.activation_loss(conv_output)
        loss = recon_loss + kl_loss + act_loss
        loss.backward()
        if self.clip_grad_norm is None:
            grad_norm = 0.
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.parameters(), self.clip_grad_norm)
        self.optimizer.step()
        opt_info.reconLoss.append(recon_loss.item())
        opt_info.klLoss.append(kl_loss.item())
        opt_info.activationLoss.append(act_loss.item())
        opt_info.gradNorm.append(grad_norm.item())
        opt_info.convActivation.append(
            conv_output[0].detach().cpu().view(-1).numpy())  # Keep 1 full one.
        return opt_info

    def vae_loss(self, samples):
        observation = samples.observation[0]  # [T,B,C,H,W]->[B,C,H,W]
        target_observation = samples.observation[self.delta_T]
        if self.delta_T > 0:
            action = samples.action[:-1]  # [T-1,B,A]  don't need the last one
            if self.onehot_action:
                action = self.distribution.to_onehot(action)
            t, b = action.shape[:2]
            action = action.transpose(1, 0)  # [B,T-1,A]
            action = action.reshape(b, -1)
        else:
            action = None
        observation, target_observation, action = buffer_to(
            (observation, target_observation, action),
            device=self.device
        )

        h, conv_out = self.encoder(observation)
        z, mu, logvar = self.vae_head(h, action)
        recon_z = self.decoder(z)

        if target_observation.dtype == torch.uint8:
            target_observation = target_observation.type(torch.float)
            target_observation = target_observation.mul_(1 / 255.)

        b, c, h, w = target_observation.shape
        recon_losses = F.binary_cross_entropy(
            input=recon_z.reshape(b * c, h, w),
            target=target_observation.reshape(b * c, h, w),
            reduction="none",
        )
        if self.delta_T > 0:
            valid = valid_from_done(samples.done).type(torch.bool)  # [T,B]
            valid = valid[-1]  # [B]
            valid = valid.to(self.device)
        else:
            valid = None  # all are valid
        recon_losses = recon_losses.view(b, c, h, w).sum(dim=(2, 3))  # sum over H,W
        recon_losses = recon_losses.mean(dim=1)  # mean over C (o/w loss is HUGE)
        recon_loss = valid_mean(recon_losses, valid=valid)  # mean over batch

        kl_losses = 1 + logvar - mu.pow(2) - logvar.exp()
        kl_losses = kl_losses.sum(dim=-1)  # sum over latent dimension
        kl_loss = -0.5 * valid_mean(kl_losses, valid=valid)  # mean over batch
        kl_loss = self.kl_coeff * kl_loss

        return recon_loss, kl_loss, conv_out

    def validation(self, itr):
        logger.log("Computing validation loss...")
        val_info = ValInfo(*([] for _ in range(len(ValInfo._fields))))
        self.optimizer.zero_grad()
        for _ in range(self.n_validation_batches):
            samples = self.replay_buffer.sample_batch(self.batch_size,
                validation=True)
            with torch.no_grad():
                recon_loss, kl_loss, conv_output = self.vae_loss(samples)
            val_info.reconLoss.append(recon_loss.item())
            val_info.klLoss.append(kl_loss.item())
            val_info.convActivation.append(
                conv_output[0].detach().cpu().view(-1).numpy())  # Keep 1 full one.
        self.optimizer.zero_grad()
        logger.log("...validation loss completed.")
        return val_info

    def state_dict(self):
        return dict(
            encoder=self.encoder.state_dict(),
            vae_head=self.vae_head.state_dict(),
            decoder=self.decoder.state_dict(),
            optimizer=self.optimizer.state_dict(),
        )

    def load_state_dict(self, state_dict):
        self.encoder.load_state_dict(state_dict["encoder"])
        self.vae_head.load_state_dict(state_dict["vae_head"])
        self.decoder.load_state_dict(state_dict["decoder"])
        self.optimizer.load_state_dict(state_dict["optimizer"])

    def parameters(self):
        yield from self.encoder.parameters()
        yield from self.vae_head.parameters()
        yield from self.decoder.parameters()

    def named_parameters(self):
        """To allow filtering by name in weight decay."""
        yield from self.encoder.named_parameters()
        yield from self.vae_head.named_parameters()
        yield from self.decoder.named_parameters()

    def eval(self):
        self.encoder.eval()  # in case of batch norm
        self.vae_head.eval()
        self.decoder.eval()

    def train(self):
        self.encoder.train()
        self.vae_head.train()
        self.decoder.train()
