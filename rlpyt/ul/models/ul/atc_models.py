
import torch

from rlpyt.models.mlp import MlpModel
from rlpyt.utils.tensor import infer_leading_dims


class ContrastModel(torch.nn.Module):

    def __init__(self, latent_size, anchor_hidden_sizes):
        super().__init__()
        if anchor_hidden_sizes is not None:
            self.anchor_mlp = MlpModel(
                input_size=latent_size,
                hidden_sizes=anchor_hidden_sizes,
                output_size=latent_size,
            )
        else:
            self.anchor_mlp = None
        self.W = torch.nn.Linear(latent_size, latent_size, bias=False)

    def forward(self, anchor, positive):
        lead_dim, T, B, _ = infer_leading_dims(anchor, 1)
        assert lead_dim == 1  # Assume [B,C] shape
        if self.anchor_mlp is not None:
            anchor = anchor + self.anchor_mlp(anchor)  # skip probably helps
        pred = self.W(anchor)
        logits = torch.matmul(pred, positive.T)
        logits = logits - torch.max(logits, dim=1, keepdim=True)[0]  # normalize
        return logits
