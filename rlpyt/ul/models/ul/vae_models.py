
import torch
import torch.nn.functional as F
import numpy as np

from rlpyt.models.mlp import MlpModel


class VaeHeadModel(torch.nn.Module):

    def __init__(self, latent_size, action_size, hidden_sizes):
        super().__init__()
        self.head = MlpModel(
            input_size=latent_size + action_size,
            hidden_sizes=hidden_sizes,
            output_size=latent_size * 2,
        )
        self._latent_size = latent_size

    def forward(self, h, action=None):
        """Assume [B] leading dimension."""
        h = F.relu(h)
        x = h if action is None else torch.cat([h, action], dim=-1)
        head = self.head(x)
        mu = head[:, :-self._latent_size]
        logvar = head[:, self._latent_size:]

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = eps * std + mu
        return z, mu, logvar


class VaeDecoderModel(torch.nn.Module):

    def __init__(
            self,
            latent_size,
            reshape,
            channels=None,
            kernel_sizes=None,
            strides=None,
            paddings=None,
            output_paddings=None,
            ):
        super().__init__()
        self.linear = torch.nn.Linear(latent_size, int(np.prod(reshape)))
        self.convt = ConvTranspose2dModel(
            in_channels=reshape[0],
            channels=channels or [32, 32, 32, 9],  # defaults for DMControl?
            kernel_sizes=kernel_sizes or [3, 3, 3, 3],
            strides=strides or [2, 2, 2, 1],
            paddings=paddings or [0, 0, 0, 0],
            output_paddings=output_paddings or [0, 1, 1, 0],
            sigmoid_output=True,
        )
        self.reshape = reshape  # [e.g. (32, 9, 9)]

    def forward(self, latent):
        """Assume [B] leading dimension."""
        x = self.linear(latent)
        b, h = x.shape
        x = x.reshape(b, *self.reshape)
        convt = self.convt(x)
        return convt


class ConvTranspose2dModel(torch.nn.Module):

    def __init__(
            self,
            in_channels,
            channels,
            kernel_sizes,
            strides,
            paddings=None,
            output_paddings=None,
            nonlinearity=torch.nn.ReLU,
            sigmoid_output=False,
            ):
        super().__init__()
        if paddings is None:
            paddings = [0 for _ in range(len(channels))]
        if output_paddings is None:
            output_paddings = [0 for _ in range(len(channels))]
        assert len(channels) == len(kernel_sizes) == len(strides) == len(paddings) == len(output_paddings)
        in_channels = [in_channels] + list(channels[:-1])
        convt_layers = [torch.nn.ConvTranspose2d(
            in_channels=ic,
            out_channels=oc,
            kernel_size=k,
            stride=s,
            padding=p,
            output_padding=op,)
            for (ic, oc, k, s, p, op) in zip(
                in_channels, channels, kernel_sizes, strides, paddings, output_paddings)]
        sequence = list()
        for convt_layer in convt_layers:
            sequence.append(convt_layer)
            sequence.append(nonlinearity())
        sequence.pop(-1)  # Remove the last nonlinearity
        if sigmoid_output:
            sequence.append(torch.nn.Sigmoid())
        self.convt = torch.nn.Sequential(*sequence)

    def forward(self, input):
        """Assumes shape is already [B,C,H,W]."""
        return self.convt(input)
