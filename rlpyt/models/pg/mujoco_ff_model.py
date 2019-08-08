
import numpy as np
import torch

from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims
from rlpyt.models.mlp import MlpModel


class MujocoFfModel(torch.nn.Module):

    def __init__(
            self,
            observation_shape,
            action_size,
            hidden_sizes=None,  # None for default (see below).
            hidden_nonlinearity=torch.nn.Tanh,  # Module form.
            mu_nonlinearity=torch.nn.Tanh,  # Module form.
            init_log_std=0.,
            ):
        super().__init__()
        self._obs_ndim = len(observation_shape)
        input_size = int(np.prod(observation_shape))
        hidden_sizes = hidden_sizes or [64, 64]
        mu_mlp = MlpModel(
            input_size=input_size,
            hidden_sizes=hidden_sizes,
            output_size=action_size,
            nonlinearity=hidden_nonlinearity,
        )
        if mu_nonlinearity is not None:
            self.mu = torch.nn.Sequential(mu_mlp, mu_nonlinearity())
        else:
            self.mu = mu_mlp
        self.v = MlpModel(
            input_size=input_size,
            hidden_sizes=hidden_sizes,
            output_size=1,
            nonlinearity=hidden_nonlinearity,
        )
        self.log_std = torch.nn.Parameter(init_log_std * torch.ones(action_size))

    def forward(self, observation, prev_action, prev_reward):
        """Feedforward layers process as [T*B,H]. Return same leading dims as
        input, can be [T,B], [B], or []."""

        # Infer (presence of) leading dimensions: [T,B], [B], or [].
        lead_dim, T, B, _ = infer_leading_dims(observation, self._obs_ndim)

        obs_flat = observation.view(T * B, -1)
        mu = self.mu(obs_flat)
        v = self.v(obs_flat).squeeze(-1)
        log_std = self.log_std.repeat(T * B, 1)

        # Restore leading dimensions: [T,B], [B], or [], as input.
        mu, log_std, v = restore_leading_dims((mu, log_std, v), lead_dim, T, B)

        return mu, log_std, v
