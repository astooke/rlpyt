
import torch

from rlpyt.algos.base import RlAlgorithm
from rlpyt.utils.quick_args import save__init__args
from rlpyt.utils.logging import logger
from rlpyt.replays.uniform import UniformReplayBuffer
from rlpyt.utils.collections import namedarraytuple

OptInfo = namedarraytuple("OptInfo",
    ["muLoss", "qLoss", "muGradNorm", "qGradNorm"])
OptData = None
SamplesToReplay = namedarraytuple("SamplesToReplay",
    ["observation", "action", "reward", "done"])


class DDPG(RlAlgorithm):

    def __init__(
            self,
            discount=0.99,
            batch_size=32,
            min_steps_learn=int(5e4),
            replay_size=int(1e6),
            training_intensity=8,  # Data_Consumption / Data_Generation
            target_update_tau=0.01,
            mu_learning_rate=1e-4,
            q_learning_rate=1e-3,
            OptimCls=torch.optim.Adam,
            optim_kwargs=None,
            initial_mu_optim_state_dict=None,
            initial_q_optim_state_dict=None,
            policy_noise=0.1,
            policy_noise_clip=None,
            grad_norm_clip=1e6,
            q_target_clip=1e6,
            policy_update_delay=1,
            ):
        if optim_kwargs is None:
            optim_kwargs = dict()
        save__init__args(locals())
        self.update_counter = 0

    def initialize(self, agent, n_itr, batch_spec, mid_batch_reset, examples):
        if agent.recurrent:
            raise NotImplementedError
        self.agent = agent
        agent.set_policy_noise(self.policy_noise, self.policy_noise_clip)
        self.n_itr = n_itr
        self.mu_optimizer = self.OptimCls(agent.mu_parameters(),
            lr=self.mu_learning_rate, **self.optim_kwargs)
        self.q_optimizer = self.OptimCls(agent.q_parameters(),
            lr=self.q_learning_rate, **self.optim_kwargs)
        if self.initial_mu_optim_state_dict is not None:
            self.mu_optimizer.load_state_dict(self.initial_mu_optim_state_dict)
        if self.initial_q_optim_state_dict is not None:
            self.q_optimizer.load_state_dict(self.initial_q_optim_state_dict)

        sample_bs = batch_spec.size
        train_bs = self.batch_size
        assert (self.training_intensity * sample_bs) % train_bs == 0
        self.updates_per_optimize = int((self.training_intensity * sample_bs) //
            train_bs)
        logger.log(f"From sampler batch size {sample_bs}, training "
            f"batch size {train_bs}, and training intensity "
            f"{self.training_intensity}, computed {self.updates_per_optimize} "
            f"updates per iteration.")

        self.min_itr_learn = self.min_steps_learn // sample_bs

        example_to_replay = SamplesToReplay(
            observation=examples["observation"],
            action=examples["action"],
            reward=examples["reward"],
            done=examples["done"],
        )
        replay_kwargs = dict(
            example=example_to_replay,
            size=self.replay_size,
            B=batch_spec.B,
        )
        self.replay_buffer = UniformReplayBuffer(**replay_kwargs)

    def optimize_agent(self, samples, itr):
        samples_to_replay = SamplesToReplay(
            observation=samples.env.observation,
            action=samples.agent.action,
            reward=samples.env.reward,
            done=samples.env.done,
        )
        self.replay_buffer.append_samples(samples_to_replay)
        if itr < self.min_itr_learn:
            return OptData(), OptInfo()  # TODO fix for empty
        opt_info = OptInfo(muLoss=[], qLoss=[], muGradNorm=[], qGradNorm=[])
        for _ in range(self.updates_per_optimize):
            self.update_counter += 1
            samples_from_replay = self.replay_buffer.sample(self.batch_size)
            self.q_optimizer.zero_grad()
            q_loss = self.q_loss(samples_from_replay)
            q_loss.backward()
            q_grad_norm = torch.nn.utils.clip_grad_norm_(
                self.agent.q_parameters(), self.clip_grad_norm)
            self.q_optimizer.step()
            opt_info.qLoss.append(q_loss.item())
            opt_info.qGradNorm.append(q_grad_norm)
            if self.update_counter % self.policy_update_delay == 0:
                self.mu_optimizer.zero_grad()
                mu_loss = self.mu_loss(samples_from_replay)
                mu_loss.backward()
                mu_grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.agent.mu_parameters(), self.clip_grad_norm)
                self.mu_optimizer.step()
                self.agent.update_target(self.target_update_tau)
                opt_info.muLoss.append(mu_loss.item())
                opt_info.muGradNorm.append(mu_grad_norm)
        return OptData(), opt_info  # TODO: fix opt_data

    def mu_loss(self, samples):
        mu_losses = self.agent.q_at_mu(*samples.agent_inputs)
        return -torch.mean(mu_losses)

    def q_loss(self, samples):
        """Samples have leading batch dimension [B,..] (but not time)."""
        q = self.agent.q(*samples.agent_inputs, samples.action)
        with torch.no_grad():
            target_q = self.agent.target_q_at_mu(*samples.next_agent_inputs)
        y = samples.reward + (1 - samples.done) * self.discount * target_q
        y = torch.clamp(y, -self.q_target_clip, self.q_target_clip)
        q_losses = 0.5 * (y - q) ** 2
        return torch.mean(q_losses)
