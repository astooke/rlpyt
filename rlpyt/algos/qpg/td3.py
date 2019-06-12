
import torch

from rlpyt.algos.dpg.ddpg import DDPG
from rlpyt.utils.quick_args import save__init__args
from rlpyt.utils.tensor import valid_mean


class TD3(DDPG):

    def __init__(
            self,
            batch_size=100,
            training_ratio=100,  # data_consumption / data_generation
            target_update_tau=0.005,
            target_update_interval=2,
            policy_update_interval=2,
            mu_learning_rate=1e-3,
            q_learning_rate=1e-3,
            **kwargs
            ):
        save__init__args(locals())
        super().__init__(**kwargs)

    def q_loss(self, samples):
        q1, q2 = self.agent.q(*samples.agent_inputs, samples.action)
        with torch.no_grad():
            target_q1, target_q2 = self.agent.target_q_at_mu(
                *samples.target_inputs)  # Includes target action noise.
            target_q = torch.min(target_q1, target_q2)
        y = samples.reward + (1 - samples.done.float()) * self.discount * target_q
        q1_losses = 0.5 * (y - q1) ** 2
        q2_losses = 0.5 * (y - q2) ** 2
        q_loss = valid_mean(q1_losses + q2_losses, samples.valid)  # valid can be None.
        return q_loss
