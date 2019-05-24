
import torch

from rlpyt.algos.base import RlAlgorithm
from rlpyt.agents.base import AgentInputs
from rlpyt.utils.quick_args import save__init__args
from rlpyt.utils.logging import logger
from rlpyt.replays.uniform import UniformReplayBuffer
from rlpyt.replays.prioritized import PrioritizedReplayBuffer


class DQN(RlAlgorithm):

    def __init__(
            self,
            discount=0.99,
            batch_size=32,
            min_steps_learn=int(5e4),
            delta_clip=1,
            replay_size=int(1e6),
            training_intensity=8,  # Average number training uses per datum.
            target_update_steps=int(1e4),
            reward_horizon=1,
            OptimCls=torch.optim.Adam,
            optim_kwargs=None,
            initial_optim_state_dict=None,
            grad_norm_clip=10.,
            eps_init=1,
            eps_final=0.01,
            eps_steps=int(1e6),
            eps_eval=0.001,
            dueling_dqn=False,
            priotized_replay=False,
            pri_alpha=0.6,
            pri_beta_init=0.5,
            pri_beta_final=1.,
            pri_beta_steps=int(50e6),
            default_priority=None,
            ):
        if optim_kwargs is None:
            opt_kwargs = dict()
        if default_priority is None:
            default_priority = delta_clip
        save__init__args(locals())

    def initialize(self, agent, n_itr, batch_spec, mid_batch_reset=False):
        self.agent = agent
        self.n_itr = n_itr
        self.optimizer = self.OptimCls(agent.parameters(),
            lr=self.learning_rate, **self.optim_kwargs)
        if self.initial_optim_state_dict is not None:
            self.optimizer.load_state_dict(self.initial_optim_state_dict)
        sample_bs = batch_spec.size
        train_bs = self.batch_size

        assert (self.training_intensity * sample_bs) % train_bs == 0
        self.updates_per_optimize = int((self.training_intensity * sample_bs) //
            train_bs)
        logger.log(f"From sampler batch size {sample_bs}, DQN "
            f"batch size {train_bs}, and training intensity "
            f"{self.training_intensity}, computed {self.updates_per_optimize} "
            f"updates per iteration.")

        self.eps_itr = max(1, self.eps_steps // sample_bs)
        self.target_update_itr = max(1, self.target_update_steps // sample_bs)
        self.min_itr_learn = self.min_steps_learn // sample_bs
        if self.prioritized_replay:
            self.pri_beta_itr = max(1, self.pri_beta_steps // sample_bs)

        replay_kwargs = dict(
            env_spec=agent.env_spec,  # TODO pass this directly?
            size=self.replay_size,
            batch_spec=batch_spec,
            reward_horizon=self.reward_horizon,
            discount=self.discount,
        )
        if self.prioritize_replay:
            replay_kwargs.update(dict(
                alpha=self.pri_alpha,
                beta_init=self.pri_beta_init,
                beta_final=self.pri_beta_final,
                beta_itr=self.pri_beta_itr),
                default_priority=self.default_priority,
            )
            ReplayCls = PrioritizedReplayBuffer
        else:
            ReplayCls = UniformReplayBuffer
        self.replay_buffer = ReplayCls(**replay_kwargs)
