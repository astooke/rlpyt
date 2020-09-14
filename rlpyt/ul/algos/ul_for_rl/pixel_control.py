

import torch
from collections import namedtuple
import pickle

from rlpyt.ul.algos.ul_for_rl.base import BaseUlAlgorithm
from rlpyt.utils.quick_args import save__init__args
from rlpyt.utils.logging import logger
from rlpyt.ul.replays.ul_for_rl_replay import UlForRlReplayBuffer
from rlpyt.utils.buffer import buffer_to
from rlpyt.ul.models.pixel_control_models import PixelControlModel
from rlpyt.ul.models.ul.encoders import EncoderModel


OptInfo = namedtuple("OptInfo", ["pcLoss", "activationLoss",
    "gradNorm", "convActivation"])
ValInfo = namedtuple("ValInfo", ["pcLoss", "convActivation"])


class PixelControl(BaseUlAlgorithm):
    """Pixel Control loss from UNREAL agent."""

    opt_info_fields = tuple(f for f in OptInfo._fields)  # copy

    def __init__(
            self,
            batch_T,
            batch_B,
            learning_rate,
            replay_filepath,
            OptimCls=torch.optim.Adam,
            optim_kwargs=None,
            initial_state_dict=None,
            clip_grad_norm=10.,
            EncoderCls=EncoderModel,
            encoder_kwargs=None,
            ReplayCls=UlForRlReplayBuffer,
            onehot_actions=True,
            activation_loss_coefficient=0.0,
            learning_rate_anneal=None,  # cosine
            learning_rate_warmup=0,  # number of updates
            PixCtlModelCls=PixelControlModel,
            pixel_control_model_kwargs=None,
            pixel_control_filename="pixel_control_80x80_4x4.pkl",  # Looks in replay path.
            validation_split=0.0,
            n_validation_batches=0,
            ):
        optim_kwargs = dict() if optim_kwargs is None else optim_kwargs
        encoder_kwargs = dict() if encoder_kwargs is None else encoder_kwargs
        pixel_control_model_kwargs = (dict() if
            pixel_control_model_kwargs is None
            else pixel_control_model_kwargs)
        save__init__args(locals())
        assert learning_rate_anneal in [None, "cosine"]
        self._replay_T = batch_T
        self.batch_size = batch_T * batch_B  # for logging

    def initialize(self, n_updates, cuda_idx=None):
        self.device = torch.device("cpu") if cuda_idx is None else torch.device(
            "cuda", index=cuda_idx)

        examples = self.load_replay()
        self.encoder = self.EncoderCls(
            image_shape=examples.observation.shape,
            latent_size=10,  # UNUSED
            **self.encoder_kwargs
        )
        self.encoder.to(self.device)

        if self.onehot_actions:
            max_act = self.replay_buffer.samples.action.max()
            self._act_dim = max_act + 1  # To use for 1-hot encoding
        else:
            # Would need to make pixel control with action as INPUT to Q-network.
            raise NotImplementedError

        channels = self.pixel_control_model_kwargs.pop("channels", [])
        channels.append(self._act_dim)
        self.pixel_control_model = self.PixCtlModelCls(
            input_shape=self.encoder.conv_out_shape,
            channels=channels,
            **self.pixel_control_model_kwargs
        )
        self.pixel_control_model.to(self.device)

        self.optim_initialize(n_updates)

        if self.initial_state_dict is not None:
            self.load_state_dict(self.initial_state_dict)

    def load_replay(self):
        if isinstance(self.replay_filepath, (list, tuple)):
            pixel_control_buffer = list()
            for filepath in self.replay_filepath:
                pc_file = filepath.rsplit("/", 1)[0] + "/" + self.pixel_control_filename
                with open(pc_file, "rb") as fh:
                    pixel_control_buffer.append(pickle.load(fh))
        else:
            pc_file = self.replay_filepath.rsplit("/", 1)[0] + "/" + self.pixel_control_filename
            with open(pc_file, "rb") as fh:
                pixel_control_buffer = pickle.load(fh)
        examples = super().load_replay(pixel_control_buffer=pixel_control_buffer)
        return examples

    def optimize(self, itr):
        opt_info = OptInfo(*([] for _ in range(len(OptInfo._fields))))
        samples = self.replay_buffer.sample_batch(self.batch_B)
        if self.lr_scheduler is not None:
            self.lr_scheduler.step(itr)  # Do every itr instead of every epoch
        self.optimizer.zero_grad()
        pc_loss, conv_output = self.pixel_control_loss(samples)
        act_loss = self.activation_loss(conv_output)
        loss = pc_loss + act_loss  # was getting a cuda vs cpu backprop error?
        loss.backward()
        if self.clip_grad_norm is None:
            grad_norm = 0.
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.parameters(), self.clip_grad_norm)
        self.optimizer.step()
        opt_info.pcLoss.append(pc_loss.item())
        opt_info.activationLoss.append(act_loss.item())
        opt_info.gradNorm.append(grad_norm.item())
        opt_info.convActivation.append(
            conv_output[0].detach().cpu().view(-1).numpy())  # Keep 1 full one.
        return opt_info

    def pixel_control_loss(self, samples):
        t, b, c, h, w = samples.observation.shape
        observation = samples.observation.view(t * b, c, h, w)
        t, b, hp, wp = samples.pixctl_return.shape
        pc_return = samples.pixctl_return.view(t * b, hp, wp)
        action = samples.action.view(t * b)

        observation, pc_return, action = buffer_to(
            (observation, pc_return, action), device=self.device)

        _, conv_output = self.encoder(observation)

        q_pc = self.pixel_control_model(conv_output)  # [B,A,H',W']
        q_pc_at_a = q_pc[torch.arange(t * b), action]  # [B,H',W']
        pc_losses = 0.5 * (q_pc_at_a - pc_return) ** 2  # [B,H',W']
        pc_losses = pc_losses.sum(dim=(1, 2))  # [B] SUM over cells
        pc_loss = pc_losses.mean()  # MEAN over batch
        return pc_loss, conv_output

    def validation(self, itr):
        logger.log("Computing validation loss...")
        val_info = ValInfo(*([] for _ in range(len(ValInfo._fields))))
        self.optimizer.zero_grad()
        for _ in range(self.n_validation_batches):
            samples = self.replay_buffer.sample_batch(self.batch_B,
                validation=True)
            with torch.no_grad():
                pc_loss, conv_output = self.pixel_control_loss(samples)
            val_info.pcLoss.append(pc_loss.item())
            val_info.convActivation.append(
                conv_output[0].detach().cpu().view(-1).numpy())  # Keep 1 full one.
        self.optimizer.zero_grad()
        logger.log("...validation loss completed.")
        return val_info

    def state_dict(self):
        return dict(
            encoder=self.encoder.state_dict(),
            pixel_control=self.pixel_control_model.state_dict(),
            optimizer=self.optimizer.state_dict(),
        )

    def load_state_dict(self, state_dict):
        self.encoder.load_state_dict(state_dict["encoder"])
        self.pixel_control_model.load_state_dict(state_dict["pixel_control"])
        self.optimizer.load_state_dict(state_dict["optimizer"])

    def parameters(self):
        yield from self.encoder.parameters()
        yield from self.pixel_control_model.parameters()

    def named_parameters(self):
        """To allow filtering by name in weight decay."""
        yield from self.encoder.named_parameters()
        yield from self.pixel_control_model.named_parameters()

    def eval(self):
        self.encoder.eval()  # in case of batch norm
        self.pixel_control_model.eval()

    def train(self):
        self.encoder.train()
        self.pixel_control_model.train()
