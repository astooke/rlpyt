
import torch

from rlpyt.models.mlp import MlpModel
from rlpyt.models.utils import conv2d_output_shape


class Conv2dModel(torch.nn.Module):

    def __init__(
            self,
            in_channels,
            channels,
            kernels,
            strides,
            paddings=None,
            nonlinearity=torch.nn.ReLU,  # Module, not Functional.
            use_maxpool=False,  # if True: convs use stride 1, maxpool downsample.
            head_sizes=None,  # Put an MLP head on top.
            ):
        super().__init__()
        if paddings is None:
            paddings = [0 for _ in range(len(channels))]
        assert len(channels) == len(kernels) == len(strides) == len(paddings)
        in_channels = [in_channels] + channels[:-1]
        ones = [1 for _ in range(len(strides))]
        if use_maxpool:
            maxp_strides = strides
            strides = ones
        else:
            maxp_strides = ones
        conv_layers = [torch.nn.Conv2d(in_channels=ic, out_channels=oc,
            kernel_size=k, stride=s, padding=p) for (ic, oc, k, s, p) in
            zip(in_channels, channels, kernels, strides, paddings)]
        sequence = list()
        for conv_layer, maxp_stride in zip(conv_layers, maxp_strides):
            sequence.extend([conv_layer, nonlinearity])
            if maxp_stride > 1:
                sequence.append(torch.nn.MaxPool2d(maxp_stride))  # No padding.
        self.conv = torch.nn.Sequential(*sequence)

    def forward(self, input):
        return self.conv(input)

    def conv_out_size(self, h, w, c=None):
        for child in self.conv.children():
            h, w = conv2d_output_shape(h, w, child.kernel_size, child.stride,
                child.padding)
            try:
                c = child.out_channels
            except AttributeError:
                pass  # MaxPool2d
        return h * w * c


class Conv2dHeadModel(Conv2dModel):

    def __init__(
            self,
            image_shape,
            channels,
            kernels,
            strides,
            hidden_sizes,
            output_size=None,
            paddings=None,
            nonlinearity=torch.nn.ReLU,
            use_maxpool=False,
            ):
        c, h, w = image_shape
        super().__init__(c, channels, kernels, strides, paddings, nonlinearity,
            use_maxpool)
        self._conv_out_size = super().conv_out_size(h, w)
        if hidden_sizes or output_size:
            self.head = MlpModel(self._conv_out_size, hidden_sizes,
                output_size=output_size, nonlinearity=nonlinearity)
            if output_size is not None:
                self._output_size = output_size
            else:
                self._output_size = (hidden_sizes if
                    isinstance(hidden_sizes, int) else hidden_sizes[-1])
        else:
            self.head = lambda x: x
            self._output_size = self._conv_out_size

    def forward(self, input):
        return self.head(self.conv(input))

    @property
    def conv_out_size(self):
        return self._conv_out_size

    @property
    def output_size(self):
        return self._output_size
