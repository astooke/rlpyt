
import torch
import torch.nn.function as F

from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims
from rlpyt.models.mlp import MlpModel


class MuMlpModel(torch.nn.Module):

    def __init__(
            self,
            observation_size,
            hidden_sizes,
            action_size,
            output_max=1,
            obs_n_dim=1,
            ):
        super().__init__()
        self._output_max = output_max
        self._obs_n_dim = obs_n_dim
        self.mlp = MlpModel(observation_size, hidden_sizes, action_size)

    def forward(self, observation, prev_action, prev_reward):
        obs_shape, T, B, has_T, has_B = infer_leading_dims(observation,
            self._obs_n_dim)
        mu = self._output_max * F.tanh(self.mlp(observation.view(T * B, -1)))
        mu = restore_leading_dims(mu, T, B, has_T, has_B)
        return mu


class PiMlpModel(torch.nn.Module):

    def __init__(
            self,
            observation_size,
            hidden_sizes,
            action_size,
            obs_n_dim=1,
            ):
        self._obs_n_dim = obs_n_dim
        self._action_size = action_size
        self.mlp = MlpModel(observation_size, hidden_sizes, action_size * 2)

    def forward(self, observation, prev_action, prev_reward):
        obs_shape, T, B, has_T, has_B = infer_leading_dims(observation,
            self._obs_n_dim)
        output = self.mlp(observation.view(T * B, -1))
        mu, log_std = output[:, :self._action_size], output[:, self._action_size:]
        mu, log_std = restore_leading_dims((mu, log_std), T, B, has_T, has_B)
        return mu, log_std


class QofMuMlpModel(torch.nn.Module):

    def __init__(
            self,
            observation_size,
            hidden_sizes,
            action_size,
            obs_n_dim=1,
            ):
        super().__init__()
        self._obs_n_dim = obs_n_dim
        self.mlp = MlpModel(observation_size + action_size, hidden_sizes, 1)

    def forward(self, observation, prev_action, prev_reward, action):
        obs_shape, T, B, has_T, has_B = infer_leading_dims(observation,
            self._obs_n_dim)
        q_input = torch.cat(
            [observation.view(T * B, -1), action.view(T * B, -1)], dim=1)
        q = self.mlp(q_input).squeeze(-1)
        q = restore_leading_dims(q, T, B, has_T, has_B)
        return q


class VMlpModel(torch.nn.Module):

    def __init__(
            self,
            observation_size,
            hidden_sizes,
            action_size=None,
            obs_n_dim=1,
            ):
        super().__init__()
        self._obs_n_dim = obs_n_dim
        self.mlp = MlpModel(observation_size, hidden_sizes, 1)

    def forward(self, observation, prev_action, prev_reward, action):
        obs_shape, T, B, has_T, has_B = infer_leading_dims(observation,
            self._obs_n_dim)
        v = self.mlp(observation.view(T * B, -1)).squeeze(-1)
        v = restore_leading_dims(v, T, B, has_T, has_B)
        return v
