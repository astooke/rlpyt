
import torch
import torch.nn.functional as F

from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims
from rlpyt.models.utils import conv2d_output_shape
from rlpyt.models.utils import scale_grad


class AtariDqnModel(torch.nn.Module):

    def __init__(
            self,
            image_shape,
            output_dim,
            n_atoms=51,
            # conv_channels,
            # conv_sizes,
            # conv_strides,
            # conv_pads,
            # pool_sizes,
            fc_size=512,
            dueling=False,
            # name="atari_cnn_lstm",
            ):
        """Should NOT run any forward code here, because cannot change torch
        num_threads after doing so, but will init before forking to worker
        processes, which might have different torch num_threads."""
        super().__init__()
        self.dueling = dueling
        self.output_dim = output_dim
        self.n_atoms = n_atoms

        # Hard-code just to get it running.
        c, h, w = image_shape  # Track image shape along with conv definition.
        self.conv1 = torch.nn.Conv2d(
            in_channels=c,
            out_channels=32,
            kernel_size=8,
            stride=1,
            padding=0,
        )
        h, w = conv2d_output_shape(h, w, kernel_size=8, stride=1, padding=0)

        self.maxp1 = torch.nn.MaxPool2d(4)
        h, w = conv2d_output_shape(h, w, kernel_size=4, stride=4, padding=0)

        self.conv2 = torch.nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=4,
            stride=1,
            padding=0,
        )
        h, w = conv2d_output_shape(h, w, kernel_size=4, stride=1, padding=0)

        self.maxp2 = torch.nn.MaxPool2d(2)
        h, w = conv2d_output_shape(h, w, kernel_size=2, stride=2, padding=0)

        self.conv3 = torch.nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=0,
        )
        h, w = conv2d_output_shape(h, w, kernel_size=3, stride=1, padding=0)

        fc_in_size = h * w * 32

        # DON'T do this in __init__().
        # test_mat = torch.zeros(1, *image_shape)
        # test_mat = self.conv1(test_mat)
        # test_mat = self.maxp1(test_mat)
        # test_mat = self.conv2(test_mat)
        # test_mat = self.maxp2(test_mat)
        # fc_in_size = test_mat.numel()

        if dueling:
            self.fc_a = torch.nn.Linear(fc_in_size, fc_size)
            self.linear_a = torch.nn.Linear(fc_size, output_dim * n_atoms, bias=False)
            self.bias_a = torch.nn.Parameter(torch.zeros(n_atoms))
            self.fc_v = torch.nn.Linear(fc_in_size, fc_size)
            self.linear_v = torch.nn.Linear(fc_size, n_atoms)
            self._head = self._dueling_head
        else:
            self.fc = torch.nn.Linear(fc_in_size, fc_size)
            self.linear_z = torch.nn.Linear(fc_size, output_dim * n_atoms)
            self._head = self._z_head

    def forward(self, image, prev_action, prev_reward):
        """Feedforward layers process as [T*B,H]. Return same leading dims as
        input, can be [T,B], [B], or []."""
        img = image.to(torch.float)  # Expect torch.uint8 inputs
        img = img.mul_(1. / 255)  # From [0-255] to [0-1], in place.

        # Infer (presence of) leading dimensions: [T,B], [B], or [].
        img_shape, T, B, has_T, has_B = infer_leading_dims(img, 3)

        img = img.view(T * B, *img_shape)  # Fold if time and batch dimensions.
        img = F.relu(self.maxp1(self.conv1(img)))
        img = F.relu(self.maxp2(self.conv2(img)))
        img = F.relu(self.conv3(img))
        flat_img = img.view(T * B, -1)

        z = self._head(flat_img)  # Dueling or not.
        z = F.softmax(z, dim=-1)  # shape: [T*B, output_dim, n_atoms]

        # Restore leading dimensions: [T,B], [B], or [], as input.
        z = restore_leading_dims(z, T, B, has_T, has_B)

        return z

    def _z_head(self, flat_img, T, B):
        fc_out = F.relu(self.fc(flat_img))
        z_flat = self.linear_z(fc_out)
        return z_flat.view(T * B, self.output_dim, self.n_atoms)

    def _dueling_head(self, flat_img, T, B):
        flat_img = scale_grad(flat_img, 2 ** (-1 / 2))
        fc_a_out = F.relu(self.fc_a(flat_img))
        adv_flat = self.linear_a(fc_a_out)
        adv = adv_flat.view(T * B, self.output_dim, self.n_atoms)
        adv += self.bias_a  # Shared across output_dim but not atoms.
        fc_v_out = F.relu(self.fc_v(flat_img))
        val_flat = self.linear_v(fc_v_out)
        val = val_flat.view(T * B, 1, self.n_atoms)
        return val + (adv - adv.mean(dim=1, keepdim=True))  # mean output_dim
