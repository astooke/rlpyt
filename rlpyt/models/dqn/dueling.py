
import torch

from rlpyt.models.mlp import MlpModel
from rlpyt.models.utils import scale_grad


class DuelingHeadModel(torch.nn.Module):

    def __init__(
            self,
            input_size,
            hidden_sizes,
            output_size,
            grad_scale=2 ** (-1 / 2),
            ):
        super().__init__()
        if isinstance(hidden_sizes, int):
            hidden_sizes = [hidden_sizes]
        self.advantage_hidden = MlpModel(input_size, hidden_sizes)
        self.advantage_out = torch.nn.Linear(hidden_sizes[-1], output_size,
            bias=False)
        self.advantage_bias = torch.nn.Parameter(torch.zeros(1))
        self.value = MlpModel(input_size, hidden_sizes, output_size=1)
        self._grad_scale = grad_scale

    def forward(self, input):
        x = scale_grad(input, self._grad_scale)
        advantage = self.advantage(x)
        value = self.value(x)
        return value + (advantage - advantage.mean(dim=-1, keepdim=True))

    def advantage(self, input):
        x = self.advantage_hidden(input)
        return self.advantage_out(x) + self.advantage_bias


class DistributionalDuelingHeadModel(torch.nn.Module):

    def __init__(
            self,
            input_size,
            hidden_sizes,
            output_size,
            n_atoms,
            grad_scale=2 ** (-1 / 2),
            ):
        super().__init__()
        if isinstance(hidden_sizes, int):
            hidden_sizes = [hidden_sizes]
        self.advantage_hidden = MlpModel(input_size, hidden_sizes)
        self.advantage_out = torch.nn.Linear(hidden_sizes[-1],
            output_size * n_atoms, bias=False)
        self.advantage_bias = torch.nn.Parameter(torch.zeros(n_atoms))
        self.value = MlpModel(input_size, hidden_sizes, output_size=n_atoms)
        self._grad_scale = grad_scale
        self._output_size = output_size
        self._n_atoms = n_atoms

    def forward(self, input):
        x = scale_grad(input, self._grad_scale)
        advantage = self.advantage(x)
        value = self.value(x).view(-1, 1, self._n_atoms)
        return value + (advantage - advantage.mean(dim=1, keepdim=True))

    def advantage(self, input):
        x = self.advantage_hidden(input)
        x = self.advantage_out(x)
        x = x.view(-1, self._output_size, self._n_atoms)
        return x + self.advantage_bias
