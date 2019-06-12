
import torch
import torch.nn.functional as F

from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims
from rlpyt.models.utils import conv2d_output_shape
from rlpyt.models.utils import scale_grad
from rlpyt.utils.collections import namedarraytuple


RnnState = namedarraytuple("RnnState", ["h", "c"])


class AtariR2d1Model(torch.nn.Module):

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
            lstm_size=512,
            lstm_layers=1,
            head_size=512,
            dueling=True,
            # name="atari_cnn_lstm",
            ):
        """Should NOT run any forward code here, because cannot change torch
        num_threads after doing so, but will init before forking to worker
        processes, which might have different torch num_threads."""
        super().__init__()
        self.dueling = dueling

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
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=0,
        )
        h, w = conv2d_output_shape(h, w, kernel_size=3, stride=1, padding=0)

        fc_in_size = h * w * 64
        self.fc_from_conv = torch.nn.Linear(fc_in_size, fc_size)
        lstm_in_size = fc_size + output_dim + 1
        self.lstm = torch.nn.LSTM(lstm_in_size, lstm_size, lstm_layers)

        # DON'T do this in __init__().
        # test_mat = torch.zeros(1, *image_shape)
        # test_mat = self.conv1(test_mat)
        # test_mat = self.maxp1(test_mat)
        # test_mat = self.conv2(test_mat)
        # test_mat = self.maxp2(test_mat)
        # fc_in_size = test_mat.numel()

        if dueling:
            self.fc_a = torch.nn.Linear(lstm_size, head_size)
            self.linear_a = torch.nn.Linear(head_size, output_dim, bias=False)
            self.bias_a = torch.nn.Parameter(torch.zeros(1))
            self.fc_v = torch.nn.Linear(lstm_size, head_size)
            self.linear_v = torch.nn.Linear(head_size, 1)
            self._head = self._dueling_head
        else:
            self.fc = torch.nn.Linear(lstm_size, head_size)
            self.linear_q = torch.nn.Linear(head_size, output_dim)
            self._head = self._q_head

    def forward(self, image, prev_action, prev_reward, init_rnn_state):
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
        fc_img = F.relu(self.fc_from_conv(img.view(T * B, -1)))

        lstm_input = torch.cat([
            fc_img.view(T, B, -1),
            prev_action.view(T, B, -1),  # Assumed onehot.
            prev_reward.view(T, B, 1),
            ], dim=2)
        if init_rnn_state is not None:  # [B,N,H] --> [N,B,H]
            init_rnn_state = tuple(torch.transpose(hc, 0, 1).contiguous()
                for hc in init_rnn_state)
        lstm_out, (hn, cn) = self.lstm(lstm_input, init_rnn_state)
        lstm_out_flat = lstm_out.view(T * B, -1)

        q = self._head(lstm_out_flat)  # Dueling or not.

        # Restore leading dimensions: [T,B], [B], or [], as input.
        q = restore_leading_dims(q, T, B, has_T, has_B)
        hn, cn = (hn.transpose(0, 1), cn.transpose(0, 1))  # --> [B,N,H]
        hn, cn = restore_leading_dims((hn, cn), B=B, put_B=has_B)  # No T.
        next_rnn_state = RnnState(h=hn, c=cn)

        return q, next_rnn_state

    def _q_head(self, lstm_out_flat):
        fc_out = F.relu(self.fc(lstm_out_flat))
        return self.linear_q(fc_out)

    def _dueling_head(self, lstm_out_flat):
        lstm_out_flat = scale_grad(lstm_out_flat, 2 ** (-1 / 2))
        fc_a_out = F.relu(self.fc_a(lstm_out_flat))
        adv = self.linear_a(fc_a_out) + self.bias_a  # Shared across output_dim.
        fc_v_out = F.relu(self.fc_v(lstm_out_flat))
        val = self.linear_v(fc_v_out)
        return val + (adv - adv.mean(dim=-1, keepdim=True))
