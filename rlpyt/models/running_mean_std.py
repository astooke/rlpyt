
import torch
import torch.distributed as dist
from rlpyt.utils.tensor import infer_leading_dims


class RunningMeanStdModel(torch.nn.Module):

    """Adapted from OpenAI baselines.  Maintains a running estimate of mean
    and variance of data along each dimension, accessible in the `mean` and
    `var` attributes.  Supports multi-GPU training by all-reducing statistics
    across GPUs."""

    def __init__(self, shape):
        super().__init__()
        self.register_buffer("mean", torch.zeros(shape))
        self.register_buffer("var", torch.ones(shape))
        self.register_buffer("count", torch.zeros(()))
        self.shape = shape

    def update(self, x):
        _, T, B, _ = infer_leading_dims(x, len(self.shape))
        x = x.view(T * B, *self.shape)
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        batch_count = T * B
        if dist.is_initialized():  # Assume need all-reduce.
            mean_var = torch.stack([batch_mean, batch_var])
            dist.all_reduce(mean_var)
            world_size = dist.get_world_size()
            mean_var /= world_size
            batch_count *= world_size
            batch_mean, batch_var = mean_var[0], mean_var[1]
        if self.count == 0:
            self.mean[:] = batch_mean
            self.var[:] = batch_var
        else:
            delta = batch_mean - self.mean
            total = self.count + batch_count
            self.mean[:] = self.mean + delta * batch_count / total
            m_a = self.var * self.count
            m_b = batch_var * batch_count
            M2 = m_a + m_b + delta ** 2 * self.count * batch_count / total
            self.var[:] = M2 / total
        self.count += batch_count
