
import torch

from rlpyt.algos.dqn.dqn import DQN
from rlpyt.utils.tensor import select_at_indexes, valid_mean
from rlpyt.algos.utils import valid_from_done


EPS = 1e-6  # (NaN-guard)


class CategoricalDQN(DQN):
    """Distributional DQN with fixed probability bins for the Q-value of each
    action, a.k.a. categorical."""

    def __init__(self, V_min=-10, V_max=10, **kwargs):
        """Standard __init__() plus Q-value limits; the agent configures
        the number of atoms (bins)."""
        super().__init__(**kwargs)
        self.V_min = V_min
        self.V_max = V_max
        if "eps" not in self.optim_kwargs:  # Assume optim.Adam
            self.optim_kwargs["eps"] = 0.01 / self.batch_size

    def initialize(self, *args, **kwargs):
        super().initialize(*args, **kwargs)
        self.agent.give_V_min_max(self.V_min, self.V_max)

    def async_initialize(self, *args, **kwargs):
        buffer = super().async_initialize(*args, **kwargs)
        self.agent.give_V_min_max(self.V_min, self.V_max)
        return buffer

    def loss(self, samples):
        """
        Computes the Distributional Q-learning loss, based on projecting the
        discounted rewards + target Q-distribution into the current Q-domain,
        with cross-entropy loss.  

        Returns loss and KL-divergence-errors for use in prioritization.
        """

        delta_z = (self.V_max - self.V_min) / (self.agent.n_atoms - 1)
        z = torch.linspace(self.V_min, self.V_max, self.agent.n_atoms)
        # Makde 2-D tensor of contracted z_domain for each data point,
        # with zeros where next value should not be added.
        next_z = z * (self.discount ** self.n_step_return)  # [P']
        next_z = torch.ger(1 - samples.done_n.float(), next_z)  # [B,P']
        ret = samples.return_.unsqueeze(1)  # [B,1]
        next_z = torch.clamp(ret + next_z, self.V_min, self.V_max)  # [B,P']

        z_bc = z.view(1, -1, 1)  # [1,P,1]
        next_z_bc = next_z.unsqueeze(1)  # [B,1,P']
        abs_diff_on_delta = abs(next_z_bc - z_bc) / delta_z
        projection_coeffs = torch.clamp(1 - abs_diff_on_delta, 0, 1)  # Most 0.
        # projection_coeffs is a 3-D tensor: [B,P,P']
        # dim-0: independent data entries
        # dim-1: base_z atoms (remains after projection)
        # dim-2: next_z atoms (summed in projection)

        with torch.no_grad():
            target_ps = self.agent.target(*samples.target_inputs)  # [B,A,P']
            if self.double_dqn:
                next_ps = self.agent(*samples.target_inputs)  # [B,A,P']
                next_qs = torch.tensordot(next_ps, z, dims=1)  # [B,A]
                next_a = torch.argmax(next_qs, dim=-1)  # [B]
            else:
                target_qs = torch.tensordot(target_ps, z, dims=1)  # [B,A]
                next_a = torch.argmax(target_qs, dim=-1)  # [B]
            target_p_unproj = select_at_indexes(next_a, target_ps)  # [B,P']
            target_p_unproj = target_p_unproj.unsqueeze(1)  # [B,1,P']
            target_p = (target_p_unproj * projection_coeffs).sum(-1)  # [B,P]
        ps = self.agent(*samples.agent_inputs)  # [B,A,P]
        p = select_at_indexes(samples.action, ps)  # [B,P]
        p = torch.clamp(p, EPS, 1)  # NaN-guard.
        losses = -torch.sum(target_p * torch.log(p), dim=1)  # Cross-entropy.

        if self.prioritized_replay:
            losses *= samples.is_weights

        target_p = torch.clamp(target_p, EPS, 1)
        KL_div = torch.sum(target_p *
            (torch.log(target_p) - torch.log(p.detach())), dim=1)
        KL_div = torch.clamp(KL_div, EPS, 1 / EPS)  # Avoid <0 from NaN-guard.

        if not self.mid_batch_reset:
            valid = valid_from_done(samples.done)
            loss = valid_mean(losses, valid)
            KL_div *= valid
        else:
            loss = torch.mean(losses)

        return loss, KL_div
