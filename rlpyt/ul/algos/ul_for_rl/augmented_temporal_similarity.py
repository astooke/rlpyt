
import torch
import torch.nn.functional as F
from collections import namedtuple
import copy

from rlpyt.models.mlp import MlpModel
from rlpyt.ul.algos.ul_for_rl.base import BaseUlAlgorithm
from rlpyt.utils.quick_args import save__init__args
from rlpyt.utils.logging import logger
from rlpyt.ul.replays.ul_for_rl_replay import UlForRlReplayBuffer
from rlpyt.utils.buffer import buffer_to
from rlpyt.algos.utils import valid_from_done
from rlpyt.utils.tensor import valid_mean
from rlpyt.models.utils import update_state_dict
from rlpyt.ul.models.ul.encoders import EncoderModel
from rlpyt.ul.algos.utils.data_augs import random_shift


OptInfo = namedtuple("OptInfo", ["atsLoss", "gradNorm",
    "convActivation", "activationLoss"])
ValInfo = namedtuple("ValInfo", ["atsLoss", "convActivation"])


class AugmentedTemporalSimilarity(BaseUlAlgorithm):
    """Similarity loss (as in BYOL) against one future time step, using a
    momentum encoder for the target."""

    opt_info_fields = tuple(f for f in OptInfo._fields)  # copy

    def __init__(
            self,
            replay_filepath,
            ReplayCls=UlForRlReplayBuffer,
            delta_T=1,
            batch_T=1,
            batch_B=256,
            learning_rate=1e-3,
            learning_rate_anneal=None,  # cosine
            learning_rate_warmup=0,  # number of updates
            OptimCls=torch.optim.Adam,
            optim_kwargs=None,
            clip_grad_norm=10.,
            target_update_tau=0.01,   # 1 for hard update
            target_update_interval=1,
            EncoderCls=EncoderModel,
            encoder_kwargs=None,
            latent_size=256,
            anchor_hidden_sizes=512,
            initial_state_dict=None,
            random_shift_prob=1.,
            random_shift_pad=4,
            activation_loss_coefficient=0.,  # rarely if ever use
            validation_split=0.0,
            n_validation_batches=0,  # usually don't do it.
            ):
        encoder_kwargs = dict() if encoder_kwargs is None else encoder_kwargs
        save__init__args(locals())
        assert learning_rate_anneal in [None, "cosine"]
        self.batch_size = batch_B * batch_T  # for logging only
        self._replay_T = batch_T + delta_T

    def initialize(self, n_updates, cuda_idx=None):
        self.device = torch.device("cpu") if cuda_idx is None else torch.device(
            "cuda", index=cuda_idx)

        examples = self.load_replay()
        self.encoder = self.EncoderCls(
            image_shape=examples.observation.shape,
            latent_size=self.latent_size,
            **self.encoder_kwargs
        )
        self.target_encoder = copy.deepcopy(self.encoder)
        self.predictor = MlpModel(
            input_size=self.latent_size,
            hidden_sizes=self.anchor_hidden_sizes,
            output_size=self.latent_size,
        )
        self.encoder.to(self.device)
        self.target_encoder.to(self.device)
        self.predictor.to(self.device)

        self.optim_initialize(n_updates)

        if self.initial_state_dict is not None:
            self.load_state_dict(self.initial_state_dict)

    def optimize(self, itr):
        opt_info = OptInfo(*([] for _ in range(len(OptInfo._fields))))
        samples = self.replay_buffer.sample_batch(self.batch_B)
        if self.lr_scheduler is not None:
            self.lr_scheduler.step(itr)  # Do every itr instead of every epoch
        self.optimizer.zero_grad()
        ats_loss, conv_output = self.ats_loss(samples)
        act_loss = self.activation_loss(conv_output)
        loss = ats_loss + act_loss
        loss.backward()
        if self.clip_grad_norm is None:
            grad_norm = 0.
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.parameters(), self.clip_grad_norm)
        self.optimizer.step()
        opt_info.atsLoss.append(ats_loss.item())
        opt_info.activationLoss.append(act_loss.item())
        opt_info.gradNorm.append(grad_norm.item())
        opt_info.convActivation.append(
            conv_output[0].detach().cpu().view(-1).numpy())  # Keep 1 full one.
        if itr % self.target_update_interval == 0:
            update_state_dict(self.target_encoder, self.encoder.state_dict(),
                self.target_update_tau)
        return opt_info

    def ats_loss(self, samples):
        anchor = (samples.observation if self.delta_T == 0 else
            samples.observation[:-self.delta_T])
        positive = samples.observation[self.delta_T:]
        t, b, c, h, w = anchor.shape
        anchor = anchor.view(t * b, c, h, w)  # Treat all T,B as separate.
        positive = positive.view(t * b, c, h, w)

        if self.random_shift_prob > 0.:
            anchor = random_shift(
                imgs=anchor,
                pad=self.random_shift_pad,
                prob=self.random_shift_prob,
            )
            positive = random_shift(
                imgs=positive,
                pad=self.random_shift_pad,
                prob=self.random_shift_prob,
            )

        anchor, positive = buffer_to((anchor, positive),
            device=self.device)

        with torch.no_grad():
            z_positive, _ = self.target_encoder(positive)
        z_anchor, conv_output = self.encoder(anchor)
        q_anchor = self.predictor(z_anchor)

        q = F.normalize(q_anchor, dim=-1, p=2)
        z = F.normalize(z_positive, dim=-1, p=2)
        ats_losses = 2. - 2 * (q * z).sum(dim=-1)  # from BYOL

        valid = valid_from_done(samples.done.type(torch.bool))
        valid = valid[self.delta_T:].reshape(-1)
        valid = valid.to(self.device)
        ats_loss = valid_mean(ats_losses, valid)

        return ats_loss, conv_output

    def validation(self, itr):
        logger.log("Computing validation loss...")
        val_info = ValInfo(*([] for _ in range(len(ValInfo._fields))))
        self.optimizer.zero_grad()
        for _ in range(self.n_validation_batches):
            samples = self.replay_buffer.sample_batch(self.batch_B,
                validation=True)
            with torch.no_grad():
                ats_loss, conv_output = self.ats_loss(samples)
            val_info.atsLoss.append(ats_loss.item())
            val_info.convActivation.append(
                conv_output[0].detach().cpu().view(-1).numpy())
        self.optimizer.zero_grad()
        logger.log("...validation loss completed.")
        return val_info

    def state_dict(self):
        return dict(
            encoder=self.encoder.state_dict(),
            target_encoder=self.target_encoder.state_dict(),
            predictor=self.predictor.state_dict(),
            optimizer=self.optimizer.state_dict(),
        )

    def load_state_dict(self, state_dict):
        self.encoder.load_state_dict(state_dict["encoder"])
        self.target_encoder.load_state_dict(state_dict["target_encoder"])
        self.predictor.load_state_dict(state_dict["predictor"])
        self.optimizer.load_state_dict(state_dict["optimizer"])

    def parameters(self):
        yield from self.encoder.parameters()
        yield from self.predictor.parameters()

    def named_parameters(self):
        """To allow filtering by name in weight decay."""
        yield from self.encoder.named_parameters()
        yield from self.predictor.named_parameters()

    def eval(self):
        self.encoder.eval()  # in case of batch norm
        self.predictor.eval()

    def train(self):
        self.encoder.train()
        self.predictor.train()
