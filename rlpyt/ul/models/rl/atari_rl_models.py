

import torch
import torch.nn.functional as F

from rlpyt.models.mlp import MlpModel
from rlpyt.models.conv2d import Conv2dModel
from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims
from rlpyt.utils.logging import logger
from rlpyt.models.running_mean_std import RunningMeanStdModel


def weight_init(m):
    if isinstance(m, (torch.nn.Linear, torch.nn.Conv2d)):
        torch.nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
        torch.nn.init.zeros_(m.bias)


class AtariPgModel(torch.nn.Module):
    """Can feed in conv and/or fc1 layer from pre-trained model, or have it
    initialize new ones (if initializing new, must provide image_shape)."""

    def __init__(
            self,
            image_shape,
            action_size,
            hidden_sizes=512,
            stop_conv_grad=False,
            channels=None,  # Defaults below.
            kernel_sizes=None,
            strides=None,
            paddings=None,
            kiaming_init=True,
            normalize_conv_out=False,
            ):
        super().__init__()
        c, h, w = image_shape
        self.conv = Conv2dModel(
            in_channels=c,
            channels=channels or [32, 64, 64],
            kernel_sizes=kernel_sizes or [8, 4, 3],
            strides=strides or [4, 2, 1],
            paddings=paddings,
        )
        self._conv_out_size = self.conv.conv_out_size(h=h, w=w)
        self.pi_v_mlp = MlpModel(
            input_size=self._conv_out_size,
            hidden_sizes=hidden_sizes,
            output_size=action_size + 1,
        )
        if kiaming_init:
            self.apply(weight_init)

        self.stop_conv_grad = stop_conv_grad
        logger.log(
            "Model stopping gradient at CONV."
            if stop_conv_grad else
            "Modeul using gradients on all parameters."
        )
        if normalize_conv_out:
            # Havent' seen this make a difference yet.
            logger.log("Model normalizing conv output across all pixels.")
            self.conv_rms = RunningMeanStdModel((1,))
            self.var_clip = 1e-6
        self.normalize_conv_out = normalize_conv_out

    def forward(self, observation, prev_action, prev_reward):
        if observation.dtype == torch.uint8:
            img = observation.type(torch.float)
            img = img.mul_(1. / 255)
        else:
            img = observation

        lead_dim, T, B, img_shape = infer_leading_dims(img, 3)
        conv = self.conv(img.view(T * B, *img_shape))

        if self.stop_conv_grad:
            conv = conv.detach()
        if self.normalize_conv_out:
            conv_var = self.conv_rms.var
            conv_var = torch.clamp(conv_var, min=self.var_clip)
            # stddev of uniform [a,b] = (b-a)/sqrt(12), 1/sqrt(12)~0.29
            # then allow [0, 10]?
            conv = torch.clamp(0.29 * conv / conv_var.sqrt(), 0, 10)

        pi_v = self.pi_v_mlp(conv.view(T * B, -1))
        pi = F.softmax(pi_v[:, :-1], dim=-1)
        v = pi_v[:, -1]

        pi, v, conv = restore_leading_dims((pi, v, conv), lead_dim, T, B)
        return pi, v, conv

    def update_conv_rms(self, observation):
        if self.normalize_conv_out:
            with torch.no_grad():
                if observation.dtype == torch.uint8:
                    img = observation.type(torch.float)
                    img = img.mul_(1. / 255)
                else:
                    img = observation
                lead_dim, T, B, img_shape = infer_leading_dims(img, 3)
                conv = self.conv(img.view(T * B, *img_shape))
                self.conv_rms.update(conv.view(-1, 1))

    def parameters(self):
        if not self.stop_conv_grad:
            yield from self.conv.parameters()
        yield from self.pi_v_mlp.parameters()

    def named_parameters(self):
        if not self.stop_conv_grad:
            yield from self.conv.named_parameters()
        yield from self.pi_v_mlp.named_parameters()

    @property
    def conv_out_size(self):
        return self._conv_out_size
    


class AtariDqnModel(torch.nn.Module):
    """Can feed in conv and/or fc1 layer from pre-trained model, or have it
    initialize new ones (if initializing new, must provide image_shape)."""

    def __init__(
            self,
            image_shape,
            action_size,
            hidden_sizes=512,
            stop_conv_grad=False,
            channels=None,  # Defaults below.
            kernel_sizes=None,
            strides=None,
            paddings=None,
            kiaming_init=True,
            normalize_conv_out=False,
            ):
        super().__init__()
        c, h, w = image_shape
        self.conv = Conv2dModel(
            in_channels=c,
            channels=channels or [32, 64, 64],
            kernel_sizes=kernel_sizes or [8, 4, 3],
            strides=strides or [4, 2, 1],
            paddings=paddings,
        )
        self._conv_out_size = self.conv.conv_out_size(h=h, w=w)
        self.q_mlp = MlpModel(
            input_size=self._conv_out_size,
            hidden_sizes=hidden_sizes,
            output_size=action_size,
        )
        if kiaming_init:
            self.apply(weight_init)

        self.stop_conv_grad = stop_conv_grad
        logger.log(
            "Model stopping gradient at CONV."
            if stop_conv_grad else
            "Modeul using gradients on all parameters."
        )
        if normalize_conv_out:
            logger.log("Model normalizing conv output across all pixels.")
            self.conv_rms = RunningMeanStdModel((1,))
            self.var_clip = 1e-6
        self.normalize_conv_out = normalize_conv_out

    def forward(self, observation, prev_action, prev_reward):
        if observation.dtype == torch.uint8:
            img = observation.type(torch.float)
            img = img.mul_(1. / 255)
        else:
            img = observation

        lead_dim, T, B, img_shape = infer_leading_dims(img, 3)
        conv = self.conv(img.view(T * B, *img_shape))

        if self.stop_conv_grad:
            conv = conv.detach()
        if self.normalize_conv_out:
            conv_var = self.conv_rms.var
            conv_var = torch.clamp(conv_var, min=self.var_clip)
            # stddev of uniform [a,b] = (b-a)/sqrt(12), 1/sqrt(12)~0.29
            # then allow [0, 10]?
            conv = torch.clamp(0.29 * conv / conv_var.sqrt(), 0, 10)

        q = self.q_mlp(conv.view(T * B, -1))

        q, conv = restore_leading_dims((q, conv), lead_dim, T, B)
        return q, conv

    def update_conv_rms(self, observation):
        if self.normalize_conv_out:
            with torch.no_grad():
                if observation.dtype == torch.uint8:
                    img = observation.type(torch.float)
                    img = img.mul_(1. / 255)
                else:
                    img = observation
                lead_dim, T, B, img_shape = infer_leading_dims(img, 3)
                conv = self.conv(img.view(T * B, *img_shape))
                self.conv_rms.update(conv.view(-1, 1))

    def parameters(self):
        if not self.stop_conv_grad:
            yield from self.conv.parameters()
        yield from self.q_mlp.parameters()

    def named_parameters(self):
        if not self.stop_conv_grad:
            yield from self.conv.named_parameters()
        yield from self.q_mlp.named_parameters()

    @property
    def conv_out_size(self):
        return self._conv_out_size
