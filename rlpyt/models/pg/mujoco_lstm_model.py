
import numpy as np
import torch

from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims
from rlpyt.models.mlp import MlpModel
from rlpyt.models.running_mean_std import RunningMeanStdModel
from rlpyt.utils.collections import namedarraytuple

RnnState = namedarraytuple("RnnState", ["h", "c"])


class MujocoLstmModel(torch.nn.Module):
    """
    Recurrent model for Mujoco locomotion agents: an MLP into an LSTM which
    outputs distribution means, log_std, and state-value estimate.
    """
    def __init__(
            self,
            observation_shape,
            action_size,
            hidden_sizes=None,  # None for default (see below).
            lstm_size=256,
            nonlinearity=torch.nn.ReLU,
            normalize_observation=False,
            norm_obs_clip=10,
            norm_obs_var_clip=1e-6,
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
        if normalize_observation:
            self.obs_rms = RunningMeanStdModel(observation_shape)
            self.norm_obs_clip = norm_obs_clip
            self.norm_obs_var_clip = norm_obs_var_clip
        self.normalize_observation = normalize_observation



    def forward(self, observation, prev_action, prev_reward, init_rnn_state):
        """
        Compute mean, log_std, and value estimate from input state. Infer
        leading dimensions of input: can be [T,B], [B], or []; provides
        returns with same leading dims.  Intermediate feedforward layers
        process as [T*B,H], and recurrent layers as [T,B,H], with T=1,B=1 when
        not given. Used both in sampler and in algorithm (both via the agent).
        Also returns the next RNN state.
        """
        # Infer (presence of) leading dimensions: [T,B], [B], or [].
        lead_dim, T, B, _ = infer_leading_dims(observation, self._obs_n_dim)

        if self.normalize_observation:
            obs_var = self.obs_rms.var
            if self.norm_obs_var_clip is not None:
                obs_var = torch.clamp(obs_var, min=self.norm_obs_var_clip)
            observation = torch.clamp((observation - self.obs_rms.mean) /
                obs_var.sqrt(), -self.norm_obs_clip, self.norm_obs_clip)

        mlp_out = self.mlp(observation.view(T * B, -1))
        lstm_input = torch.cat([
            mlp_out.view(T, B, -1),
            prev_action.view(T, B, -1),
            prev_reward.view(T, B, 1),
            ], dim=2)
        init_rnn_state = None if init_rnn_state is None else tuple(init_rnn_state)
        lstm_out, (hn, cn) = self.lstm(lstm_input, init_rnn_state)
        outputs = self.head(lstm_out.view(T * B, -1))
        mu = outputs[:, :self._action_size]
        log_std = outputs[:, self._action_size:-1]
        v = outputs[:, -1]

        # Restore leading dimensions: [T,B], [B], or [], as input.
        mu, log_std, v = restore_leading_dims((mu, log_std, v), lead_dim, T, B)
        # Model should always leave B-dimension in rnn state: [N,B,H]
        next_rnn_state = RnnState(h=hn, c=cn)

        return mu, log_std, v, next_rnn_state

    def update_obs_rms(self, observation):
        if self.normalize_observation:
            self.obs_rms.update(observation)
