
import torch

from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims
from rlpyt.utils.collections import namedarraytuple
from rlpyt.models.conv2d import Conv2dHeadModel
from rlpyt.models.mlp import MlpModel
from rlpyt.models.dqn.dueling import DuelingHeadModel


RnnState = namedarraytuple("RnnState", ["h", "c"])


class AtariR2d1Model(torch.nn.Module):
    """2D convolutional neural network (for multiple video frames per
    observation) feeding into an LSTM and MLP output for Q-value outputs for
    the action set."""
    def __init__(
            self,
            image_shape,
            output_size,
            fc_size=512,  # Between conv and lstm.
            lstm_size=512,
            head_size=512,
            dueling=False,
            use_maxpool=False,
            channels=None,  # None uses default.
            kernel_sizes=None,
            strides=None,
            paddings=None,
            ):
        """Instantiates the neural network according to arguments; network defaults
        stored within this method."""
        super().__init__()
        self.dueling = dueling
        self.conv = Conv2dHeadModel(
            image_shape=image_shape,
            channels=channels or [32, 64, 64],
            kernel_sizes=kernel_sizes or [8, 4, 3],
            strides=strides or [4, 2, 1],
            paddings=paddings or [0, 1, 1],
            use_maxpool=use_maxpool,
            hidden_sizes=fc_size,  # ReLU applied here (Steven).
        )
        self.lstm = torch.nn.LSTM(self.conv.output_size + output_size + 1, lstm_size)
        if dueling:
            self.head = DuelingHeadModel(lstm_size, head_size, output_size)
        else:
            self.head = MlpModel(lstm_size, head_size, output_size=output_size)

    def forward(self, observation, prev_action, prev_reward, init_rnn_state):
        """Feedforward layers process as [T*B,H]. Return same leading dims as
        input, can be [T,B], [B], or []."""
        img = observation.type(torch.float)  # Expect torch.uint8 inputs
        img = img.mul_(1. / 255)  # From [0-255] to [0-1], in place.

        # Infer (presence of) leading dimensions: [T,B], [B], or [].
        lead_dim, T, B, img_shape = infer_leading_dims(img, 3)

        conv_out = self.conv(img.view(T * B, *img_shape))  # Fold if T dimension.

        lstm_input = torch.cat([
            conv_out.view(T, B, -1),
            prev_action.view(T, B, -1),  # Assumed onehot.
            prev_reward.view(T, B, 1),
            ], dim=2)
        init_rnn_state = None if init_rnn_state is None else tuple(init_rnn_state)
        lstm_out, (hn, cn) = self.lstm(lstm_input, init_rnn_state)

        q = self.head(lstm_out.view(T * B, -1))

        # Restore leading dimensions: [T,B], [B], or [], as input.
        q = restore_leading_dims(q, lead_dim, T, B)
        # Model should always leave B-dimension in rnn state: [N,B,H].
        next_rnn_state = RnnState(h=hn, c=cn)

        return q, next_rnn_state
