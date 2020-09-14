
import torch

from rlpyt.ul.models.mlp import MlpModel
from rlpyt.utils.tensor import infer_leading_dims


class InverseModel(torch.nn.Module):

    def __init__(
            self,
            input_size,
            hidden_sizes,
            action_size,
            num_actions,
            subtract=False,
            use_input="conv",  # ["conv", "z"]
        ):
        super().__init__()
        if use_input != "conv":
            raise NotImplementedError
        self.mlp = MlpModel(
            input_size=2 * input_size,
            hidden_sizes=hidden_sizes,
            output_size=action_size * num_actions,
        )
        self._action_size = action_size
        self._num_actions = num_actions
        self._use_input = use_input
        self._subtract = subtract

    def forward(self, conv_obs, conv_last):
        lead_dim, T, B, _ = infer_leading_dims(conv_obs, 3)
        assert lead_dim == 1  # has [B], not [B,T]
        obs = conv_obs.view(B, -1)
        last = conv_last.view(B, -1)
        if self._subtract:
            last = last - obs
        mlp_input = torch.cat([obs, last], dim=-1)
        act_logits = self.mlp(mlp_input)
        act_logits = act_logits.view(B, self._num_actions, self._action_size)
        return act_logits
