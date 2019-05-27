
import torch

from rlpyt.algos.q_learning.dqn import DQN


class CategoricalDQN(DQN):

    def __init__(self, V_min=-10, V_max=10, **kwargs):
        super().__init__(**kwargs)
        if "eps" not in self.optim_kwargs:  # Assume optim.Adam
            self.optim_kwargs["eps"] = 0.01 / self.batch_size

    def loss(self, samples):
        if self.agent.recurrent:
            raise NotImplementedError
        ps = self.agent(*samples.agent_inputs)
        z = torch.linspace(self.V_min, self.V_max, self.agent.n_atoms)
        z_contract = (self.discount ** self.n_step_return) * z
        # Makde 2-D tensor of contracted z_domain for each data point,
        # with zeros where next value should not be added.
        z_contract_notdone = torch.ger(1 - samples.done_n, z_contract)
        ret = samples.return_.unsqueeze(1)
        z_next = torch.clamp(ret + z_contract_notdone, self.V_min, self.V_max)
        # TODO: finish this.

        delta_z = (self.V_max - self.V_min) / (self.agent.n_atoms - 1)