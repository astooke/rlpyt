
import torch

from rlpyt.models.conv2d import Conv2dModel
from rlpyt.models.mlp import MlpModel
from rlpyt.ul.models.dmlab_conv2d import DmlabConv2dModel
from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims


def weight_init(m):
    """Kaiming_normal is standard for relu networks, sometimes."""
    if isinstance(m, (torch.nn.Linear, torch.nn.Conv2d)):
        torch.nn.init.kaiming_normal_(m.weight, mode="fan_in", 
            nonlinearity="relu")
        torch.nn.init.zeros_(m.bias)


class EncoderModel(torch.nn.Module):

    def __init__(
            self,
            image_shape,
            latent_size,
            channels,
            kernel_sizes,
            strides,
            paddings=None,
            hidden_sizes=None,  # usually None; NOT the same as anchor MLP
            kiaming_init=True,
            ):
        super().__init__()
        c, h, w = image_shape
        self.conv = Conv2dModel(
            in_channels=c,
            channels=channels,
            kernel_sizes=kernel_sizes,
            strides=strides,
            paddings=paddings,
            use_maxpool=False,
        )
        self._output_size = self.conv.conv_out_size(h, w)
        self._output_shape = self.conv.conv_out_shape(h, w)
        self.head = MlpModel(
            input_size=self._output_size,
            hidden_sizes=hidden_sizes,
            output_size=latent_size,
        )
        if kiaming_init:
            self.apply(weight_init)

    def forward(self, observation):
        lead_dim, T, B, img_shape = infer_leading_dims(observation, 3)
        if observation.dtype == torch.uint8:
            img = observation.type(torch.float)
            img = img.mul_(1. / 255)
        else:
            img = observation
        conv = self.conv(img.view(T * B, *img_shape))
        c = self.head(conv.view(T * B, -1))

        c, conv = restore_leading_dims((c, conv), lead_dim, T, B)

        return c, conv  # In case wanting to do something with conv output

    @property
    def output_size(self):
        return self._output_size

    @property
    def output_shape(self):
        return self._output_shape


class DmlabEncoderModel(torch.nn.Module):

    def __init__(
            self,
            image_shape,
            latent_size,
            use_fourth_layer=True,
            skip_connections=True,
            hidden_sizes=None,
            kiaming_init=True,
            ):
        super().__init__()
        c, h, w = image_shape
        self.conv = DmlabConv2dModel(
            in_channels=c,
            use_fourth_layer=True,
            skip_connections=skip_connections,
            use_maxpool=False,
        )
        self._output_size = self.conv.output_size(h, w)
        self._output_shape = self.conv.output_shape(h, w)

        self.head = MlpModel(  # gets to z_t, not necessarily c_t
            input_size=self._output_size,
            hidden_sizes=hidden_sizes,
            output_size=latent_size,
        )
        if kiaming_init:
            self.apply(weight_init)

    def forward(self, observation):
        lead_dim, T, B, img_shape = infer_leading_dims(observation, 3)
        if observation.dtype == torch.uint8:
            img = observation.type(torch.float)
            img = img.mul_(1. / 255)
        else:
            img = observation
        conv = self.conv(img.view(T * B, *img_shape))
        c = self.head(conv.view(T * B, -1))

        c, conv = restore_leading_dims((c, conv), lead_dim, T, B)

        return c, conv  # In case wanting to do something with conv output

    @property
    def output_size(self):
        return self._output_size

    @property
    def output_shape(self):
        return self._output_shape
