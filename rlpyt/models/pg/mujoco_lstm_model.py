
import numpy as np
import torch

from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims
from rlpyt.models.mlp import MlpModel
from rlpyt.utils.collections import namedarraytuple

RnnState = namedarraytuple("RnnState", ["h", "c"])


class MujocoLstmModel(torch.nn.Module):

    def __init__(
            self,
            observation_shape,
            action_size,
            hidden_sizes=None,  # None for default (see below).
            lstm_size=256,
            nonlinearity=torch.nn.ReLU,
            ):
        super().__init__()
        self._obs_n_dim = len(observation_shape)
        self._action_size = action_size
        hidden_sizes = hidden_sizes or [256, 256]
        mlp_input_size = int(np.prod(observation_shape))
        self.mlp = MlpModel(
            input_size=mlp_input_size,
            hidden_sizes=hidden_sizes,
            output_size=None,
            nonlinearity=nonlinearity,
        )
        mlp_output_size = hidden_sizes[-1] if hidden_sizes else mlp_input_size
        self.lstm = torch.nn.LSTM(mlp_output_size + action_size + 1, lstm_size)
        self.head = torch.nn.Linear(lstm_size, action_size * 2 + 1)

    def forward(self, observation, prev_action, prev_reward, init_rnn_state):
        """Feedforward layers process as [T*B,H]. Return same leading dims as
        input, can be [T,B], [B], or []."""

        # Infer (presence of) leading dimensions: [T,B], [B], or [].
        lead_dim, T, B, _ = infer_leading_dims(observation, self._obs_n_dim)

        mlp_out = self.mlp(observation.view(T * B, -1))
        lstm_input = torch.cat([
            mlp_out.view(T, B, -1),
            prev_action.view(T, B, -1),
            prev_reward.view(T, B, 1),
            ], dim=2)
        init_rnn_state = None if init_rnn_state is None else tuple(init_rnn_state)
        lstm_out, (hn, cn) = self.lstm(lstm_input, init_rnn_state)
        outputs = self.head(lstm_out)
        mu = outputs[:, :self._action_size]
        log_std = outputs[:, self._action_size:-1]
        v = outputs[:, -1].squeeze(-1)

        # Restore leading dimensions: [T,B], [B], or [], as input.
        mu, log_std, v = restore_leading_dims((mu, log_std, v), lead_dim, T, B)
        # Model should always leave B-dimension in rnn state: [N,B,H]
        next_rnn_state = RnnState(h=hn, c=cn)

        return mu, log_std, v, next_rnn_state
