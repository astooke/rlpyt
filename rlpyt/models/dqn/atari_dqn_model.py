
import torch

from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims
from rlpyt.models.conv2d import Conv2dModel
from rlpyt.models.mlp import MlpModel
from rlpyt.models.dqn.dueling import DuelingHeadModel


class AtariDqnModel(torch.nn.Module):

    def __init__(
            self,
            image_shape,
            output_size,
            fc_sizes=512,
            dueling=False,
            use_maxpool=False,
            channels=None,  # None uses default.
            kernel_sizes=None,
            strides=None,
            paddings=None,
            ):
        super().__init__()
        self.dueling = dueling
        c, h, w = image_shape
        self.conv = Conv2dModel(
            in_channels=c,
            channels=channels or [32, 64, 64],
            kernel_sizes=kernel_sizes or [8, 4, 3],
            strides=strides or [4, 2, 1],
            paddings=paddings or [0, 1, 1],
            use_maxpool=use_maxpool,
        )
        conv_out_size = self.conv.conv_out_size(h, w)
        if dueling:
            self.head = DuelingHeadModel(conv_out_size, fc_sizes, output_size)
        else:
            self.head = MlpModel(conv_out_size, fc_sizes, output_size)

    def forward(self, observation, prev_action, prev_reward):
        """Feedforward layers process as [T*B,H]. Return same leading dims as
        input, can be [T,B], [B], or []."""
        img = observation.type(torch.float)  # Expect torch.uint8 inputs
        img = img.mul_(1. / 255)  # From [0-255] to [0-1], in place.

        # Infer (presence of) leading dimensions: [T,B], [B], or [].
        lead_dim, T, B, img_shape = infer_leading_dims(img, 3)

        conv_out = self.conv(img.view(T * B, *img_shape))  # Fold if T dimension.
        q = self.head(conv_out.view(T * B, -1))

        # Restore leading dimensions: [T,B], [B], or [], as input.
        q = restore_leading_dims(q, lead_dim, T, B)
        return q
