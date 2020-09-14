

import torch
import torch.nn as nn
import torch.nn.functional as F

from rlpyt.models.conv2d import Conv2dModel
from rlpyt.models.mlp import MlpModel
from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims


def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        # delta-orthogonal init from https://arxiv.org/pdf/1806.05393.pdf
        assert m.weight.size(2) == m.weight.size(3)
        m.weight.data.fill_(0.0)
        m.bias.data.fill_(0.0)
        mid = m.weight.size(2) // 2
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)


class SacModel(nn.Module):
    """To keep the standard agent.model interface for shared params, etc."""

    def __init__(self, conv, pi_fc1, pi_mlp):
        super().__init__()
        self.conv = conv
        self.pi_fc1 = pi_fc1
        self.pi_mlp = pi_mlp

    def forward(self, observation, prev_action, prev_reward):
        """Just to keep the standard obs, prev_action, prev_rew interface."""
        conv = self.conv(observation)
        latent = self.pi_fc1(conv)
        mu, log_std = self.pi_mlp(latent, prev_action, prev_reward)
        return mu, log_std, latent, conv


class SacConvModel(nn.Module):

    def __init__(
            self,
            image_shape,
            channels=None,
            kernel_sizes=None,
            strides=None,
            paddings=None,
            final_nonlinearity=True,
            ):
        super().__init__()
        c, h, w = image_shape
        self.conv = Conv2dModel(
            in_channels=c,
            channels=channels or [32, 32, 32, 32],
            kernel_sizes=kernel_sizes or [3, 3, 3, 3],
            strides=strides or [2, 1, 1, 1],
            paddings=paddings,
            final_nonlinearity=final_nonlinearity,
        )
        self._output_shape = self.conv.conv_out_shape(h=h, w=w, c=c)
        self._output_size = self.conv.conv_out_size(h=h, w=w, c=c)

    def forward(self, observation):
        if observation.dtype == torch.uint8:
            img = observation.type(torch.float)
            img = img.mul_(1. / 255)
        else:
            img = observation

        lead_dim, T, B, img_shape = infer_leading_dims(img, 3)
        conv = self.conv(img.view(T * B, *img_shape))
        conv = restore_leading_dims(conv, lead_dim, T, B)
        return conv

    @property
    def output_shape(self):
        return self._output_shape

    @property
    def output_size(self):
        return self._output_size


class SacFc1Model(nn.Module):

    def __init__(
            self,
            input_size,
            latent_size,
            layer_norm=True,
            ):
        super().__init__()
        self.linear = nn.Linear(input_size, latent_size)
        self.layer_norm = nn.LayerNorm(latent_size) if layer_norm else None
        self._output_size = latent_size

    def forward(self, conv_out):
        if conv_out.dtype == torch.uint8:  # Testing NoConv model
            conv_out = conv_out.type(torch.float)
            conv_out = conv_out.mul_(1. / 255)
        lead_dim, T, B, _ = infer_leading_dims(conv_out, 3)
        conv_out = F.relu(conv_out.view(T * B, -1))  # bc conv_out might be pre-activation
        latent = self.linear(conv_out)
        if self.layer_norm is not None:
            latent = self.layer_norm(latent)
        latent = restore_leading_dims(latent, lead_dim, T, B)
        return latent

    @property
    def output_size(self):
        return self._output_size


class SacActorModel(nn.Module):

    def __init__(
            self,
            input_size,
            action_size,
            hidden_sizes,
            min_log_std=-10.,
            max_log_std=2.,
            ):
        super().__init__()
        self.mlp = MlpModel(
            input_size=input_size,
            hidden_sizes=hidden_sizes,
            output_size=action_size * 2,
        )
        self.apply(weight_init)
        self.min_log_std = min_log_std
        self.max_log_std = max_log_std

    def forward(self, latent, prev_action=None, prev_reward=None):
        lead_dim, T, B, _ = infer_leading_dims(latent, 1)  # latent is vector

        out = self.mlp(latent.view(T * B, -1))
        mu, log_std = out.chunk(chunks=2, dim=-1)
        # Squash log_std into range.
        log_std = torch.tanh(log_std)
        log_std = self.min_log_std + 0.5 * (
            self.max_log_std - self.min_log_std) * (1 + log_std)
        mu, log_std = restore_leading_dims((mu, log_std), lead_dim, T, B)
        return mu, log_std


class SacCriticModel(nn.Module):

    def __init__(
            self,
            input_size,
            action_size,
            hidden_sizes,
            ):
        super().__init__()
        self.mlp1 = MlpModel(
            input_size=input_size + action_size,
            hidden_sizes=hidden_sizes,
            output_size=1,
        )
        self.mlp2 = MlpModel(
            input_size=input_size + action_size,
            hidden_sizes=hidden_sizes,
            output_size=1,
        )
        self.apply(weight_init)

    def forward(self, latent, action, prev_action=None, prev_reward=None):
        lead_dim, T, B, _ = infer_leading_dims(latent, 1)  # latent is vector

        q_input = torch.cat([
            latent.view(T * B, -1),
            action.view(T * B, -1),
            ], dim=1)
        q1 = self.mlp1(q_input).squeeze(-1)
        q2 = self.mlp2(q_input).squeeze(-1)
        q1, q2 = restore_leading_dims((q1, q2), lead_dim, T, B)
        return q1, q2


class SacNoConvModel(nn.Module):
    """To keep the standard agent.model interface for shared params, etc.

    RESULT: yeah this didn't work in most envs, except a bit in walker.
    """

    def __init__(self, pi_fc1, pi_mlp):
        super().__init__()
        # self.conv = conv
        self.pi_fc1 = pi_fc1
        self.pi_mlp = pi_mlp

    def forward(self, observation, prev_action, prev_reward):
        """Just to keep the standard obs, prev_action, prev_rew interface."""
        # conv = self.conv(observation)
        conv = observation
        latent = self.pi_fc1(conv)
        mu, log_std = self.pi_mlp(latent, prev_action, prev_reward)
        return mu, log_std, latent
