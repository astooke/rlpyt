
import torch


class MlpModel(torch.nn.Module):
    """Multilayer Perceptron with last layer linear."""

    def __init__(
            self,
            input_size,
            hidden_sizes,  # Can be empty list for none.
            output_size=None,  # if None, last layer has nonlinearity applied.
            nonlinearity=torch.nn.ReLU,  # Module, not Functional.
            ):
        super().__init__()
        if isinstance(hidden_sizes, int):
            hidden_sizes = [hidden_sizes]
        hidden_layers = [torch.nn.Linear(n_in, n_out) for n_in, n_out in
            zip([input_size] + hidden_sizes[:-1], hidden_sizes)]
        sequence = list()
        for layer in hidden_layers:
            sequence.extend([layer, nonlinearity()])
        if output_size is not None:
            last_size = hidden_sizes[-1] if hidden_sizes else input_size
            sequence.append(torch.nn.Linear(last_size, output_size))
        self.model = torch.nn.Sequential(*sequence)
        self._output_size = (hidden_sizes[-1] if output_size is None
            else output_size)

    def forward(self, input):
        return self.model(input)

    @property
    def output_size(self):
        return self._output_size
