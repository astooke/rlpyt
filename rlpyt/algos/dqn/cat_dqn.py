
import torch

from rlpyt.algos.q_learning.dqn import DQN
from rlpyt.utils.tensor import select_at_indexes, valid_mean


EPS = 1e-6  # (NaN-guard)


class CategoricalDQN(DQN):

    def __init__(self, V_min=-10, V_max=10, **kwargs):
        super().__init__(**kwargs)
        self.V_min = V_min
        self.V_max = V_max
        if "eps" not in self.optim_kwargs:  # Assume optim.Adam
            self.optim_kwargs["eps"] = 0.01 / self.batch_size

    def initialize(self, agent, *args, **kwargs):
        super().initialize(agent, *args, **kwargs)
        agent.give_V_min_max(self.V_min, self.V_max)

    def loss(self, samples):
        """Samples have leading batch dimension [B,..] (but not time)."""
        delta_z = (self.V_max - self.V_min) / (self.agent.n_atoms - 1)
        z = torch.linspace(self.V_min, self.V_max, self.agent.n_atoms)
        # Makde 2-D tensor of contracted z_domain for each data point,
        # with zeros where next value should not be added.
        next_z = z * (self.discount ** self.n_step_return)
        next_z = torch.ger(1 - samples.done_n, next_z)  # Outer product.
        ret = samples.return_.unsqueeze(1)
        next_z = torch.clamp(ret + next_z, self.V_min, self.V_max)

        z_bc = z.view(1, 1, -1)  # For broadcasting.
        next_z_bc = next_z.unsqueeze(-1)
        abs_diff_on_delta = abs(next_z_bc - z_bc) / delta_z
        projection_coeffs = torch.clamp(1 - abs_diff_on_delta, 0, 1)  # Most 0.
        # projection_coeffs is a 3-D tensor:
        # dim-0: independent data entries (looped in batched_dot)
        # dim-1: next_z atoms (summed in batched_dot)
        # dim-2: base z atoms (dim-1 after batched_dot)

        # TODO: check this whole paragraph, esp dot() and batched_dot()
        with torch.no_grad():
            target_ps = self.agent.target_p(*samples.next_agent_inputs)
            if self.double_dqn:
                next_ps = self.agent(*samples.next_agent_inputs)
                next_qs = torch.dot(next_ps, z)
                next_a = torch.argmax(next_qs, dim=-1)
            else:
                target_qs = torch.dot(target_ps, z)
                next_a = torch.argmax(target_qs, dim=-1)
            target_p = select_at_indexes(next_a, target_ps)
            target_p = torch.batched_dot(target_p, projection_coeffs)
        ps = self.agent(*samples.agent_inputs)
        p = select_at_indexes(samples.action, ps)
        p = torch.clamp(p, EPS, 1)  # NaN-guard.
        losses = -torch.sum(target_p * torch.log(p), dim=1)  # Cross-entropy.

        if self.prioritized_replay:
            losses *= samples.is_weights

        target_p = torch.clamp(target_p, EPS, 1)
        KL_div = torch.sum(target_p *
            (torch.log(target_p) - torch.log(p)), dim=1)
        KL_div = torch.clamp(KL_div, EPS, 1 / EPS)  # Avoid <0 from NaN-guard.

        if not self.mid_batch_reset:
            valid = samples.valid.type(losses.dtype)  # Convert to float.
            loss = valid_mean(losses, valid)
            KL_div *= valid
        else:
            loss = torch.mean(losses)

        return loss, KL_div
