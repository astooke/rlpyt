
import torch

from rlpyt.algos.qpg.ddpg import DDPG
from rlpyt.utils.quick_args import save__init__args
from rlpyt.utils.tensor import valid_mean


class TD3(DDPG):
    """Twin delayed deep deterministic policy gradient algorithm."""

    def __init__(
            self,
            batch_size=100,
            replay_ratio=100,  # data_consumption / data_generation
            target_update_tau=0.005,
            target_update_interval=2,
            policy_update_interval=2,
            mu_learning_rate=1e-3,
            q_learning_rate=1e-3,
            **kwargs
            ):
        """Saved input arguments."""
        super().__init__(**kwargs)
        self._batch_size = batch_size
        del batch_size  # Property.
        save__init__args(locals(), overwrite=True)

    def initialize(self, *args, **kwargs):
        super().initialize(*args, **kwargs)
        self.agent.give_min_itr_learn(self.min_itr_learn)

    def async_initialize(self, *args, **kwargs):
        ret = super().async_initialize(*args, **kwargs)
        self.agent.give_min_itr_learn(self.min_itr_learn)
        return ret

    def q_loss(self, samples, valid):
        """Computes MSE Q-loss for twin Q-values and min of target-Q values."""
        q1, q2 = self.agent.q(*samples.agent_inputs, samples.action)
        with torch.no_grad():
            target_q1, target_q2 = self.agent.target_q_at_mu(
                *samples.target_inputs)  # Includes target action noise.
            target_q = torch.min(target_q1, target_q2)
        disc = self.discount ** self.n_step_return
        y = samples.return_ + (1 - samples.done_n.float()) * disc * target_q
        q1_losses = 0.5 * (y - q1) ** 2
        q2_losses = 0.5 * (y - q2) ** 2
        q_loss = valid_mean(q1_losses + q2_losses, valid)  # valid can be None.
        return q_loss
