
import torch
import torch.nn.functional as F

from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims


class AtariFfModel(torch.nn.Module):

    def __init__(
            self,
            image_shape,
            output_dim,
            # conv_channels,
            # conv_sizes,
            # conv_strides,
            # conv_pads,
            # pool_sizes,
            fc_size=512,
            # name="atari_cnn_lstm",
            ):
        super().__init__()

        # Hard-code just to get it running.
        self.conv1 = torch.nn.Conv2d(
            in_channels=image_shape[0],
            out_channels=16,
            kernel_size=8,
            stride=1,
            padding=0,
        )
        self.maxp1 = torch.nn.MaxPool2d(2)
        self.conv2 = torch.nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=4,
            stride=1,
            padding=0,
        )
        self.maxp2 = torch.nn.MaxPool2d(2)

        test_mat = torch.zeros(1, *image_shape)
        test_mat = self.conv1(test_mat)
        test_mat = self.maxp1(test_mat)
        test_mat = self.conv2(test_mat)
        test_mat = self.maxp2(test_mat)
        # fc_in_size = test_mat.numel() + output_dim + 1
        fc_in_size = test_mat.numel()

        self.fc1 = torch.nn.Linear(fc_in_size, fc_size)
        self.linear_pi = torch.nn.Linear(fc_size, output_dim)
        self.linear_v = torch.nn.Linear(fc_size, 1)

    def forward(self, image, _prev_action, _prev_reward):
        img = image.to(torch.float)  # Expect torch.uint8 inputs
        img = img.mul_(1. / 255)  # From [0-255] to [0-1], in place.

        # Infer (presence of) leading dimensions: [T,B], [B], or [].
        img_shape, T, B, _T, _B = infer_leading_dims(img, 3)

        img = img.view(T * B, *img_shape)  # Fold time and batch dimensions.
        img = F.relu(self.maxp1(self.conv1(img)))
        img = F.relu(self.maxp2(self.conv2(img)))

        # onehot_action = self.env_spec.action_space.to_onehot(prev_action)
        fc_out = F.relu(self.fc1(img.view(T * B, -1)))
        pi = F.softmax(self.linear_pi(fc_out), dim=-1)
        v = self.linear_v(fc_out).squeeze(-1)

        # Restore leading dimensions: [T,B], [B], or [], as input.
        pi, v = restore_leading_dims((pi, v), T, B, _T, _B)

        return pi, v
