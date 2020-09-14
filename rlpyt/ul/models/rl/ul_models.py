
import torch

from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims


class UlEncoderModel(torch.nn.Module):

    def __init__(self, conv, latent_size, conv_out_size):
        super().__init__()
        self.conv = conv  # Get from RL agent's model.
        self.head = torch.nn.Linear(conv_out_size, latent_size)

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
        return c, conv
