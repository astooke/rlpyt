
import copy
import torch
from collections import OrderedDict

from rlpyt.agents.base import BaseAgent, AgentStep
from rlpyt.utils.quick_args import save__init__args
from rlpyt.ul.models.rl.sac_rl_models import (SacModel,
    SacConvModel, SacFc1Model, SacActorModel, SacCriticModel)
from rlpyt.distributions.gaussian import Gaussian, DistInfoStd
from rlpyt.utils.buffer import buffer_to
from rlpyt.models.utils import update_state_dict
from rlpyt.utils.collections import namedarraytuple
from rlpyt.utils.logging import logger


AgentInfo = namedarraytuple("AgentInfo", ["dist_info", "conv"])


class SacAgent(BaseAgent):

    def __init__(
            self,
            ModelCls=SacModel,
            ConvModelCls=SacConvModel,
            Fc1ModelCls=SacFc1Model,
            PiModelCls=SacActorModel,
            QModelCls=SacCriticModel,
            conv_kwargs=None,
            fc1_kwargs=None,
            pi_model_kwargs=None,
            q_model_kwargs=None,
            initial_state_dict=None,
            action_squash=1.,
            pretrain_std=0.75,  # 0.75 gets pretty uniform squashed actions
            load_conv=False,
            load_all=False,
            state_dict_filename=None,
            store_latent=False,
            ):
        if conv_kwargs is None:
            conv_kwargs = dict()
        if fc1_kwargs is None:
            fc1_kwargs = dict(latent_size=50)  # default
        if pi_model_kwargs is None:
            pi_model_kwargs = dict(hidden_sizes=[1024, 1024])  # default
        if q_model_kwargs is None:
            q_model_kwargs = dict(hidden_sizes=[1024, 1024])  # default
        save__init__args(locals())
        super().__init__(ModelCls=SacModel)
        self.min_itr_learn = 0  # Get from algo.
        assert not (load_conv and load_all)

    def initialize(self, env_spaces, share_memory=False,
            global_B=1, env_ranks=None):
        self.conv = self.ConvModelCls(
            image_shape=env_spaces.observation.shape,
            **self.conv_kwargs)
        self.q_fc1 = self.Fc1ModelCls(
            input_size=self.conv.output_size,
            **self.fc1_kwargs)
        self.pi_fc1 = self.Fc1ModelCls(
            input_size=self.conv.output_size,
            **self.fc1_kwargs)

        latent_size = self.q_fc1.output_size
        action_size = env_spaces.action.shape[0]

        # These are just MLPs
        self.pi_mlp = self.PiModelCls(
            input_size=latent_size,
            action_size=action_size,
            **self.pi_model_kwargs)
        self.q_mlps = self.QModelCls(
            input_size=latent_size,
            action_size=action_size,
            **self.q_model_kwargs)
        self.target_q_mlps = copy.deepcopy(self.q_mlps)  # Separate params.

        # Make reference to the full actor model including encoder.
        # CAREFUL ABOUT TRAIN MODE FOR LAYER NORM IF CHANGING THIS?
        self.model = SacModel(conv=self.conv, pi_fc1=self.pi_fc1,
            pi_mlp=self.pi_mlp)

        if self.load_conv:
            logger.log("Agent loading state dict: " + self.state_dict_filename)
            loaded_state_dict = torch.load(self.state_dict_filename,
                map_location=torch.device("cpu"))
            # From UL, saves snapshot: params["algo_state_dict"]["encoder"]
            if "algo_state_dict" in loaded_state_dict:
                loaded_state_dict = loaded_state_dict
            loaded_state_dict = loaded_state_dict.get("algo_state_dict", loaded_state_dict)
            loaded_state_dict = loaded_state_dict.get("encoder", loaded_state_dict)
            # A bit onerous, but ensures that state dicts match:
            conv_state_dict = OrderedDict([(k, v)  # .replace("conv.", "", 1)
                for k, v in loaded_state_dict.items() if k.startswith("conv.")])
            self.conv.load_state_dict(conv_state_dict)
            # Double check it gets into the q_encoder as well.
            logger.log("Agent loaded CONV state dict.")
        elif self.load_all:
            # From RL, saves snapshot: params["agent_state_dict"]
            loaded_state_dict = torch.load(self.state_dict_filename,
                map_location=torch.device('cpu'))
            self.load_state_dict(loaded_state_dict["agent_state_dict"])
            logger.log("Agnet loaded FULL state dict.")            
        else:
            logger.log("Agent NOT loading state dict.")

        self.target_conv = copy.deepcopy(self.conv)
        self.target_q_fc1 = copy.deepcopy(self.q_fc1)

        if share_memory:
            # The actor model needs to share memory to sampler workers, and
            # this includes handling the encoder!
            # (Almost always just run serial anyway, no sharing.)
            self.model.share_memory()
            self.shared_model = self.model
        if self.initial_state_dict is not None:
            raise NotImplementedError
        self.env_spaces = env_spaces
        self.share_memory = share_memory

        assert len(env_spaces.action.shape) == 1
        self.distribution = Gaussian(
            dim=env_spaces.action.shape[0],
            squash=self.action_squash,
            # min_std=np.exp(MIN_LOG_STD),  # NOPE IN PI_MODEL NOW
            # max_std=np.exp(MAX_LOG_STD),
        )

    def to_device(self, cuda_idx=None):
        super().to_device(cuda_idx)  # Takes care of self.model only.
        self.conv.to(self.device)  # should already be done
        self.q_fc1.to(self.device)
        self.pi_fc1.to(self.device)  # should already be done
        self.q_mlps.to(self.device)
        self.pi_mlp.to(self.device)  # should already be done
        self.target_conv.to(self.device)
        self.target_q_fc1.to(self.device)
        self.target_q_mlps.to(self.device)

    def give_min_itr_learn(self, min_itr_learn):
        self.min_itr_learn = min_itr_learn  # From algo.

    @torch.no_grad()
    def step(self, observation, prev_action, prev_reward):
        observation, prev_action, prev_reward = buffer_to(
            (observation, prev_action, prev_reward),
            device=self.device)
        # self.model includes encoder + actor MLP.
        mean, log_std, latent, conv = self.model(observation, prev_action, prev_reward)
        dist_info = DistInfoStd(mean=mean, log_std=log_std)
        action = self.distribution.sample(dist_info)
        agent_info = AgentInfo(dist_info=dist_info,
            conv=conv if self.store_latent else None)
        action, agent_info = buffer_to((action, agent_info), device="cpu")
        return AgentStep(action=action, agent_info=agent_info)

    def q(self, conv_out, prev_action, prev_reward, action):
        """Compute twin Q-values for state/observation and input action
        (with grad).
        Assume variables already on device."""
        latent = self.q_fc1(conv_out)
        q1, q2 = self.q_mlps(latent, action, prev_action, prev_reward)
        return q1.cpu(), q2.cpu()

    def target_q(self, conv_out, prev_action, prev_reward, action):
        """Compute twin target Q-values for state/observation and input
        action.
        Assume variables already on device."""
        latent = self.target_q_fc1(conv_out)
        target_q1, target_q2 = self.target_q_mlps(latent, action, prev_action,
            prev_reward)
        return target_q1.cpu(), target_q2.cpu()

    def pi(self, conv_out, prev_action, prev_reward):
        """Compute action log-probabilities for state/observation, and
        sample new action (with grad).  Uses special ``sample_loglikelihood()``
        method of Gaussian distriution, which handles action squashing
        through this process.
        Assume variables already on device."""
        # Call just the actor mlp, not the encoder.
        latent = self.pi_fc1(conv_out)
        mean, log_std = self.pi_mlp(latent, prev_action, prev_reward)
        dist_info = DistInfoStd(mean=mean, log_std=log_std)
        action, log_pi = self.distribution.sample_loglikelihood(dist_info)
        # action = self.distribution.sample(dist_info)
        # log_pi = self.distribution.log_likelihood(action, dist_info)
        log_pi, dist_info = buffer_to((log_pi, dist_info), device="cpu")
        return action, log_pi, dist_info  # Action stays on device for q models.

    def train_mode(self, itr):
        super().train_mode(itr)  # pi_encoder in here in model
        self.conv.train()  # should already be done
        self.q_fc1.train()
        self.pi_fc1.train()  # should already be done
        self.q_mlps.train()
        self.pi_mlp.train()  # should already be done

    def sample_mode(self, itr):
        super().sample_mode(itr)  # pi_encoder in here in model
        self.conv.eval()  # should already be done
        self.q_fc1.eval()
        self.pi_fc1.eval()  # should already be done
        self.q_mlps.eval()  # not used anyway
        self.pi_mlp.eval()  # should already be done
        if itr == 0:
            logger.log(f"Agent at itr {itr}, sample std: {self.pretrain_std}")
        if itr == self.min_itr_learn:
            logger.log(f"Agent at itr {itr}, sample std: learned.")
        std = None if itr >= self.min_itr_learn else self.pretrain_std
        self.distribution.set_std(std)  # If None: std from policy dist_info.

    def eval_mode(self, itr):
        super().eval_mode(itr)  # pi_encoder in here in model
        self.conv.eval()  # should already be done
        self.q_fc1.eval()
        self.pi_fc1.eval()  # should already be done
        self.q_mlps.eval()  # not used anyway
        self.pi_mlp.eval()  # should already be done
        self.distribution.set_std(0.)  # Deterministic (dist_info std ignored).

    def state_dict(self):
        return dict(
            conv=self.conv.state_dict(),
            q_fc1=self.q_fc1.state_dict(),
            pi_fc1=self.pi_fc1.state_dict(),
            q_mlps=self.q_mlps.state_dict(),
            pi_mlp=self.pi_mlp.state_dict(),
            target_conv=self.target_conv.state_dict(),
            target_q_fc1=self.target_q_fc1.state_dict(),
            target_q_mlps=self.target_q_mlps.state_dict(),
        )

    def load_state_dict(self, state_dict):
        self.conv.load_state_dict(state_dict["conv"])
        self.q_fc1.load_state_dict(state_dict["q_fc1"])
        self.pi_fc1.load_state_dict(state_dict["pi_fc1"])
        self.q_mlps.load_state_dict(state_dict["q_mlps"])
        self.pi_mlp.load_state_dict(state_dict["pi_mlp"])
        self.target_conv.load_state_dict(state_dict["target_conv"])
        self.target_q_fc1.load_state_dict(state_dict["target_q_fc1"])
        self.target_q_mlps.load_state_dict(state_dict["target_q_mlps"])

    def data_parallel(self, *args, **kwargs):
        raise NotImplementedError  # Do it later.

    def async_cpu(self, *args, **kwargs):
        raise NotImplementedError  # Double check this...

    def update_targets(self, q_tau=1, encoder_tau=1):
        """Do each parameter ONLY ONCE."""
        update_state_dict(self.target_conv, self.conv.state_dict(), encoder_tau)
        update_state_dict(self.target_q_fc1, self.q_fc1.state_dict(), encoder_tau)
        update_state_dict(self.target_q_mlps, self.q_mlps.state_dict(), q_tau)
