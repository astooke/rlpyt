

import torch
from collections import namedtuple
import copy

from rlpyt.ul.algos.ul_for_rl.base import BaseUlAlgorithm
from rlpyt.utils.quick_args import save__init__args
from rlpyt.utils.logging import logger
from rlpyt.ul.replays.ul_for_rl_replay import UlForRlReplayBuffer
from rlpyt.utils.buffer import buffer_to
from rlpyt.algos.utils import valid_from_done
# from rlpyt.utils.tensor import to_onehot
from rlpyt.models.utils import update_state_dict
from rlpyt.ul.models.ul.stdim_models import (StDimEncoderModel,
    StDimGlobalLocalContrastModel, StDimLocalLocalContrastModel)
from rlpyt.ul.models.ul.atc_models import ContrastModel


IGNORE_INDEX = -100  # Mask CPC samples across episode boundary.
OptInfo = namedtuple("OptInfo", ["stdimLoss", "ggLoss", "glLoss", "llLoss",
    "ggAccuracy", "glAccuracy", "llAccuracy",
    "activationLoss", "gradNorm", "convActivation"])
ValInfo = namedtuple("ValInfo", ["stdimLoss", "ggLoss", "glLoss", "llLoss",
    "ggAccuracy", "glAccuracy", "llAccuracy",
    "convActivation"])


class STDIM(BaseUlAlgorithm):
    """Spatio-Temporal Deep InfoMax, with momentum encoder for target."""

    opt_info_fields = tuple(f for f in OptInfo._fields)  # copy

    def __init__(
            self,
            replay_filepath,
            learning_rate,
            batch_B=64,
            batch_T=1,
            delta_T=1,
            use_global_global=False,
            use_global_local=True,
            use_local_local=True,
            local_conv_layer=1,  # 0-based indexing
            latent_size=256,
            target_update_tau=0.01,   # 1 for hard update
            target_update_interval=1,
            OptimCls=torch.optim.Adam,
            optim_kwargs=None,
            initial_state_dict=None,
            clip_grad_norm=100.,
            EncoderCls=StDimEncoderModel,
            encoder_kwargs=None,
            ReplayCls=UlForRlReplayBuffer,
            anchor_hidden_sizes=512,
            activation_loss_coefficient=0.0,
            learning_rate_anneal=None,  # cosine
            learning_rate_warmup=0,  # number of updates
            validation_split=0.0,
            n_validation_batches=0,
            ):
        optim_kwargs = dict() if optim_kwargs is None else optim_kwargs
        encoder_kwargs = dict() if encoder_kwargs is None else encoder_kwargs
        save__init__args(locals())
        self.c_e_loss = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
        assert learning_rate_anneal in [None, "cosine"]
        self._replay_T = batch_T + delta_T
        self.batch_size = batch_B * batch_T  # for logging

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
        self.encoder.to(self.device)
        self.target_encoder.to(self.device)

        local_size = self.encoder.conv_out_shapes[self.local_conv_layer][0]  # C from [C,H,W]
        if self.use_global_global:
            self.gg_contrast = ContrastModel(
                latent_size=self.latent_size,
                anchor_hidden_sizes=self.anchor_hidden_sizes,
            )
            self.gg_contrast.to(self.device)
        if self.use_global_local:
            self.gl_contrast = StDimGlobalLocalContrastModel(
                latent_size=self.latent_size,
                anchor_hidden_sizes=self.anchor_hidden_sizes,
                local_size=local_size,
            )
            self.gl_contrast.to(self.device)
        if self.use_local_local:
            self.ll_contrast = StDimLocalLocalContrastModel(
                anchor_hidden_sizes=self.anchor_hidden_sizes,
                local_size=local_size,
            )
            self.ll_contrast.to(self.device)

        self.optim_initialize(n_updates)

        if self.initial_state_dict is not None:
            self.load_state_dict(self.initial_state_dict)

    def optimize(self, itr):
        opt_info = OptInfo(*([] for _ in range(len(OptInfo._fields))))
        samples = self.replay_buffer.sample_batch(self.batch_size)
        if self.lr_scheduler is not None:
            self.lr_scheduler.step(itr)  # Do every itr instead of every epoch
        self.optimizer.zero_grad()
        stdim_loss, loss_vals, accuracies, conv_output = self.stdim_loss(samples)
        act_loss = self.activation_loss(conv_output)
        loss = stdim_loss + act_loss
        loss.backward()
        if self.clip_grad_norm is None:
            grad_norm = 0.
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.parameters(), self.clip_grad_norm)
        self.optimizer.step()
        opt_info.stdimLoss.append(stdim_loss.item())
        opt_info.ggLoss.append(loss_vals[0])
        opt_info.glLoss.append(loss_vals[1])
        opt_info.llLoss.append(loss_vals[2])
        opt_info.ggAccuracy.append(accuracies[0])
        opt_info.glAccuracy.append(accuracies[1])
        opt_info.llAccuracy.append(accuracies[2])
        opt_info.activationLoss.append(act_loss.item())
        opt_info.gradNorm.append(grad_norm.item())
        opt_info.convActivation.append(
            conv_output[0].detach().cpu().view(-1).numpy())  # Keep 1 full one.
        if itr % self.target_update_interval == 0:
            update_state_dict(self.target_encoder, self.encoder.state_dict(),
                self.target_update_tau)
        return opt_info

    def stdim_loss(self, samples):
        """Contrast over the batch dimension for every location, but not across
        locations."""
        anchor = samples.observation[:-self.delta_T]
        positive = samples.observation[self.delta_T:]
        t, b, c, h, w = anchor.shape
        anchor = anchor.view(t * b, c, h, w)  # Treat all T,B as separate.
        positive = positive.view(t * b, c, h, w)

        anchor, positive = buffer_to((anchor, positive),
            device=self.device)

        with torch.no_grad():
            c_positive, _, positive_conv_layers = self.target_encoder(positive)
        c_anchor, anchor_conv_out, anchor_conv_layers = self.encoder(anchor)

        labels = torch.arange(c_anchor.shape[0],   # batch size
            dtype=torch.long, device=self.device)
        valid = valid_from_done(samples.done).type(torch.bool)
        valid = valid[self.delta_T:].reshape(-1)
        labels[~valid] = IGNORE_INDEX

        gg_loss, gl_loss, ll_loss = [torch.tensor(0., device=self.device)] * 3
        gg_accuracy, gl_accuracy, ll_accuracy = [torch.tensor(0., device=self.device)] * 3
        if self.use_global_global:
            gg_logits = self.gg_contrast(c_anchor, c_positive)
            gg_loss = self.c_e_loss(gg_logits, labels)
            gg_correct = torch.argmax(gg_logits.detach(), dim=1) == labels
            gg_accuracy = torch.mean(gg_correct[valid].float())
        if self.use_global_local:
            positive_local = positive_conv_layers[self.local_conv_layer]
            gl_logits_list = self.gl_contrast(c_anchor, positive_local)
            gl_losses = torch.stack([self.c_e_loss(gl_logits, labels)
                for gl_logits in gl_logits_list])
            gl_loss = torch.mean(gl_losses)
            gl_corrects = [torch.argmax(gl_logits.detach(), dim=1) == labels
                for gl_logits in gl_logits_list]
            gl_accuracies = [torch.mean(gl_correct[valid].float())
                for gl_correct in gl_corrects]
            gl_accuracy = torch.mean(torch.stack(gl_accuracies))
        if self.use_local_local:
            anchor_local = anchor_conv_layers[self.local_conv_layer]
            positive_local = positive_conv_layers[self.local_conv_layer]
            ll_logits_list = self.ll_contrast(anchor_local, positive_local)
            ll_losses = torch.stack([self.c_e_loss(ll_logits, labels)
                for ll_logits in ll_logits_list])
            ll_loss = torch.mean(ll_losses)
            ll_corrects = [torch.argmax(ll_logits.detach(), dim=1) == labels
                for ll_logits in ll_logits_list]
            ll_accuracies = [torch.mean(ll_correct[valid].float())
                for ll_correct in ll_corrects]
            ll_accuracy = torch.mean(torch.stack(ll_accuracies))
        stdim_loss = gg_loss + gl_loss + ll_loss

        loss_vals = (gg_loss.item(), gl_loss.item(), ll_loss.item())
        accuracies = (gg_accuracy.item(), gl_accuracy.item(), ll_accuracy.item())
        return stdim_loss, loss_vals, accuracies, anchor_conv_out

    def validation(self, itr):
        logger.log("Computing validation loss...")
        val_info = ValInfo(*([] for _ in range(len(ValInfo._fields))))
        self.optimizer.zero_grad()
        for _ in range(self.n_validation_batches):
            samples = self.replay_buffer.sample_batch(self.batch_B,
                validation=True)
            with torch.no_grad():
                stdim_loss, loss_vals, accuracies, conv_output = self.stdim_loss(samples)
            val_info.stdimLoss.append(stdim_loss.item())
            val_info.ggLoss.append(loss_vals[0])
            val_info.glLoss.append(loss_vals[1])
            val_info.llLoss.append(loss_vals[2])
            val_info.ggAccuracy.append(accuracies[0])
            val_info.glAccuracy.append(accuracies[1])
            val_info.llAccuracy.append(accuracies[2])
            val_info.convActivation.append(
                conv_output[0].detach().cpu().view(-1).numpy())  # Keep 1 full one.

        self.optimizer.zero_grad()
        logger.log("...validation loss completed.")
        return val_info

    def state_dict(self):
        state_dict = dict(
            encoder=self.encoder.state_dict(),
            target_encoder=self.target_encoder.state_dict(),
            optimizer=self.optimizer.state_dict(),
        )
        if self.use_global_global:
            state_dict["gg_contrast"] = self.gg_contrast.state_dict()
        if self.use_global_local:
            state_dict["gl_contrast"] = self.gl_contrast.state_dict()
        if self.use_local_local:
            state_dict["ll_contrast"] = self.ll_contrast.state_dict()
        return state_dict

    def load_state_dict(self, state_dict):
        self.encoder.load_state_dict(state_dict["encoder"])
        self.target_encoder.load_state_dict(state_dict["target_encoder"])
        self.optimizer.load_state_dict(state_dict["optimizer"])
        if self.use_global_global:
            self.gg_contrast.load_state_dict(state_dict["gg_contrast"])
        if self.use_global_local:
            self.gl_contrast.load_state_dict(state_dict["gl_contrast"])
        if self.use_local_local:
            self.ll_contrast.load_state_dict(state_dict["ll_contrast"])

    def parameters(self):
        yield from self.encoder.parameters()
        if self.use_global_global:
            yield from self.gg_contrast.parameters()
        if self.use_global_local:
            yield from self.gl_contrast.parameters()
        if self.use_local_local:
            yield from self.ll_contrast.parameters()

    def named_parameters(self):
        """To allow filtering by name in weight decay."""
        yield from self.encoder.named_parameters()
        if self.use_global_global:
            yield from self.gg_contrast.named_parameters()
        if self.use_global_local:
            yield from self.gl_contrast.named_parameters()
        if self.use_local_local:
            yield from self.ll_contrast.named_parameters()

    def eval(self):
        self.encoder.eval()  # in case of batch norm
        if self.use_global_global:
            self.gg_contrast.eval()
        if self.use_global_local:
            self.gl_contrast.eval()
        if self.use_local_local:
            self.ll_contrast.eval()

    def train(self):
        self.encoder.train()
        if self.use_global_global:
            self.gg_contrast.train()
        if self.use_global_local:
            self.gl_contrast.train()
        if self.use_local_local:
            self.ll_contrast.train()
