

import torch
import torch.nn.functional as F

from rlpyt.models.mlp import MlpModel
from rlpyt.ul.models.dmlab_conv2d import DmlabConv2dModel
from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims
from rlpyt.utils.logging import logger
from rlpyt.utils.collections import namedarraytuple


RnnState = namedarraytuple("RnnState", ["h", "c"])  # For downstream namedarraytuples to work


def weight_init(m):
    if isinstance(m, (torch.nn.Linear, torch.nn.Conv2d)):
        torch.nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
        torch.nn.init.zeros_(m.bias)


class DmlabPgLstmModel(torch.nn.Module):

    def __init__(
            self,
            image_shape,
            output_size,
            lstm_size,
            skip_connections=True,
            hidden_sizes=None,
            kiaming_init=True,
            stop_conv_grad=False,
            skip_lstm=True,
            ):
        super().__init__()
        c, h, w = image_shape
        self.conv = DmlabConv2dModel(
            in_channels=c,
            use_fourth_layer=True,
            use_maxpool=False,
            skip_connections=skip_connections,
        )
        self._conv_out_size = self.conv.output_size(h=h, w=w)
        self.fc1 = torch.nn.Linear(
            in_features=self._conv_out_size,
            out_features=lstm_size,
        )
        self.lstm = torch.nn.LSTM(lstm_size + output_size + 1, lstm_size)
        self.pi_v_head = MlpModel(
            input_size=lstm_size,
            hidden_sizes=hidden_sizes,
            output_size=output_size + 1,
        )
        if kiaming_init:
            self.apply(weight_init)
        self.stop_conv_grad = stop_conv_grad
        logger.log(
            "Model stopping gradient at CONV."
            if stop_conv_grad else
            "Modeul using gradients on all parameters."
        )
        self._skip_lstm = skip_lstm

    def forward(self, observation, prev_action, prev_reward, init_rnn_state):
        if observation.dtype == torch.uint8:
            img = observation.type(torch.float)
            img = img.mul_(1. / 255)
        else:
            img = observation

        lead_dim, T, B, img_shape = infer_leading_dims(img, 3)
        conv = self.conv(img.view(T * B, *img_shape))

        if self.stop_conv_grad:
            conv = conv.detach()

        fc1 = F.relu(self.fc1(conv.view(T * B, -1)))
        lstm_input = torch.cat([
            fc1.view(T, B, -1),
            prev_action.view(T, B, -1),  # Assumed onehot
            prev_reward.view(T, B, 1),
            ], dim=2)
        init_rnn_state = None if init_rnn_state is None else tuple(init_rnn_state)
        lstm_out, (hn, cn) = self.lstm(lstm_input, init_rnn_state)
        if self._skip_lstm:
            lstm_out = lstm_out.view(T * B, -1) + fc1
        pi_v = self.pi_v_head(lstm_out.view(T * B, - 1))
        pi = F.softmax(pi_v[:, :-1], dim=-1)
        v = pi_v[:, -1]
        pi, v, conv = restore_leading_dims((pi, v, conv), lead_dim, T, B)
        next_rnn_state = RnnState(h=hn, c=cn)
        return pi, v, next_rnn_state, conv

    def parameters(self):
        if not self.stop_conv_grad:
            yield from self.conv.parameters()
        yield from self.fc1.parameters()
        yield from self.lstm.parameters()
        yield from self.pi_v_head.parameters()

    def named_parameters(self):
        if not self.stop_conv_grad:
            yield from self.conv.named_parameters()
        yield from self.fc1.named_parameters()
        yield from self.lstm.named_parameters()
        yield from self.pi_v_head.named_parameters()

    @property
    def conv_out_size(self):
        return self._conv_out_size


