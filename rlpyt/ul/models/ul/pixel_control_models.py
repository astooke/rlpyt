

import torch
import numpy as np

from rlpyt.models.mlp import MlpModel
from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims


class PixelControlModel(torch.nn.Module):

    def __init__(
            self,
            input_shape,
            fc_sizes,
            reshape,
            channels,
            kernel_sizes,
            strides,
            paddings=None,
            output_paddings=None,
            dueling=True,
            ):
        super().__init__()
        if isinstance(input_shape, int):
            input_shape = (input_shape,)
        self.input_shape = input_shape
        input_size = int(np.prod(input_shape))
        if fc_sizes is None:
            self.mlp = None
            if reshape is None:  # Then receiving conv input [C,H,W]
                assert len(input_shape) == 3
                in_channels = input_shape[0]
            else:
                assert input_size == int(np.prod(reshape))
        else:
            self.mlp = MlpModel(
                input_size=input_size,
                hidden_sizes=fc_sizes,
            )
            assert self.mlp.output_size == int(np.prod(reshape))
            in_channels = reshape[0]
        self.reshape = reshape
        self.dueling = dueling
        if dueling:
            channels[-1] = channels[-1] + 1  # Add a Value channel (+ Advantages)
        self.convt = ConvTranspose2dModel(
            in_channels=in_channels,
            channels=channels,
            kernel_sizes=kernel_sizes,
            strides=strides,
            paddings=paddings,
            output_paddings=output_paddings,
            sigmoid_output=False,
        )

    def forward(self, input):
        lead_dim, T, B, in_shape = infer_leading_dims(input, len(self.input_shape))
        x = input.view(T * B, *in_shape)
        if self.mlp is not None:
            x = self.mlp(input.view(T * B, -1))
        if self.reshape is not None:
            x = x.view(T * B, *self.reshape)
        x = self.convt(x)
        if self.dueling:  # then x is shaped: [T*B,A+1,H,W]
            value = x[:, :1]  # zeroth channel  [T*B,1,H,W]
            advantage = x[:, 1:]  # other channels [T*B,A,H,W]
            x = value + (advantage - advantage.mean(dim=1, keepdim=True))
        x = restore_leading_dims(x, lead_dim, T, B)
        return x


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
        in_channels = [in_channels] + channels[:-1]
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
