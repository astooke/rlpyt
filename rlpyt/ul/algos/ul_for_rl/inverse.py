
import torch
import torch.nn.functional as F
from collections import namedtuple

from rlpyt.ul.algos.ul_for_rl.base import BaseUlAlgorithm
from rlpyt.ul.models.inv_models import InverseModel
from rlpyt.utils.quick_args import save__init__args
from rlpyt.utils.logging import logger
from rlpyt.ul.replays.ul_for_rl_replay import UlForRlReplayBuffer
from rlpyt.utils.buffer import buffer_to
from rlpyt.algos.utils import valid_from_done
from rlpyt.ul.models.ul.encoders import EncoderModel
from rlpyt.ul.algos.utils.data_augs import random_shift
from rlpyt.distributions.categorical import Categorical, DistInfo


IGNORE_INDEX = -100  # Mask action samples across episode boundary.
OptInfo = namedtuple("OptInfo", [
    "invLoss", "entLoss", "accuracy", "perplexity",
    "activationLoss", "gradNorm", "convActivation"])
ValInfo = namedtuple("ValInfo", [
    "invLoss", "entLoss", "accuracy", "perplexity",
    "convActivation"])


class Inverse(BaseUlAlgorithm):
    """Inverse training to predict {a_t, a_t+1,..} from (o_t, o_t+k)."""

    opt_info_fields = tuple(f for f in OptInfo._fields)  # copy

    def __init__(
            self,
            batch_size,
            learning_rate,
            replay_filepath,
            delta_T=1,
            OptimCls=torch.optim.Adam,
            optim_kwargs=None,
            initial_state_dict=None,
            clip_grad_norm=10.,
            EncoderCls=EncoderModel,
            encoder_kwargs=None,
            ReplayCls=UlForRlReplayBuffer,
            onehot_actions=True,
            activation_loss_coefficient=0.0,
            learning_rate_anneal=None,  # cosine
            learning_rate_warmup=0,  # number of updates
            random_shift_prob=0.,
            random_shift_pad=4,
            InverseModelCls=InverseModel,
            inverse_model_kwargs=None,
            entropy_loss_coeff=0.01,
            validation_split=0.0,
            n_validation_batches=0,
            ):
        optim_kwargs = dict() if optim_kwargs is None else optim_kwargs
        encoder_kwargs = dict() if encoder_kwargs is None else encoder_kwargs
        inverse_model_kwargs = dict() if inverse_model_kwargs is None else inverse_model_kwargs
        save__init__args(locals())
        self.c_e_loss = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
        assert learning_rate_anneal in [None, "cosine"]
        assert onehot_actions  # needs discrete action space for now.
        assert delta_T > 0
        self._replay_T = delta_T + 1

    def initialize(self, n_updates, cuda_idx=None):
        self.device = torch.device("cpu") if cuda_idx is None else torch.device(
            "cuda", index=cuda_idx)

        examples = self.load_replay()
        self.encoder = self.EncoderCls(
            image_shape=examples.observation.shape,
            latent_size=10,  # UNUSED
            **self.encoder_kwargs
        )

        if self.onehot_actions:
            act_dim = self.replay_buffer.samples.action.max() + 1
            self.distribution = Categorical(act_dim)
        else:
            assert len(self.replay_buffer.samples.action.shape == 3)
            act_dim = self.replay_buffer.samples.action.shape[2]
        self.inverse_model = self.InverseModelCls(
            input_size=self.encoder.conv_out_size,
            action_size=act_dim,
            num_actions=self.delta_T,
            use_input="conv",
            **self.inverse_model_kwargs
        )
        self.encoder.to(self.device)
        self.inverse_model.to(self.device)

        self.optim_initialize(n_updates)

        if self.initial_state_dict is not None:
            self.load_state_dict(self.initial_state_dict)

    def optimize(self, itr):
        opt_info = OptInfo(*([] for _ in range(len(OptInfo._fields))))
        samples = self.replay_buffer.sample_batch(self.batch_size)
        if self.lr_scheduler is not None:
            self.lr_scheduler.step(itr)  # Do every itr instead of every epoch
        self.optimizer.zero_grad()
        inv_loss, ent_loss, accuracy, perplexity, conv_output = self.inverse_loss(samples)
        act_loss = self.activation_loss(conv_output)
        loss = inv_loss + ent_loss + act_loss
        loss.backward()
        if self.clip_grad_norm is None:
            grad_norm = 0.
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.parameters(), self.clip_grad_norm)
        self.optimizer.step()
        opt_info.invLoss.append(inv_loss.item())
        opt_info.entLoss.append(ent_loss.item())
        opt_info.accuracy.append(accuracy.item())
        opt_info.perplexity.append(perplexity.item())
        opt_info.activationLoss.append(act_loss.item())
        opt_info.gradNorm.append(grad_norm.item())
        opt_info.convActivation.append(
            conv_output[0].detach().cpu().view(-1).numpy())  # Keep 1 full one.
        return opt_info

    def inverse_loss(self, samples):
        observation = samples.observation[0]  # [T,B,C,H,W]->[B,C,H,W]
        last_observation = samples.observation[-1]

        if self.random_shift_prob > 0.:
            observation = random_shift(
                imgs=observation,
                pad=self.random_shift_pad,
                prob=self.random_shift_prob,
            )
            last_observation = random_shift(
                imgs=last_observation,
                pad=self.random_shift_pad,
                prob=self.random_shift_prob,
            )

        action = samples.action  # [T,B,A]
        # if self.onehot_actions:
        #     action = to_onehot(action, self._act_dim, dtype=torch.float)
        observation, last_observation, action = buffer_to(
            (observation, last_observation, action),
            device=self.device)

        _, conv_obs = self.encoder(observation)
        _, conv_last = self.encoder(last_observation)

        valid = valid_from_done(samples.done).type(torch.bool)  # [T,B]
        # All timesteps invalid if the last_observation is:
        valid = valid[-1].repeat(self.delta_T, 1).transpose(1, 0)  # [B,T-1]
        
        if self.onehot_actions:
            logits = self.inverse_model(conv_obs, conv_last)  # [B,T-1,A]
            labels = action[:-1].transpose(1, 0)  # [B,T-1], not the last action
            labels[~valid] = IGNORE_INDEX

            b, t, a = logits.shape
            logits = logits.view(b * t, a)
            labels = labels.reshape(b * t)
            logits = logits - torch.max(logits, dim=1, keepdim=True)[0]
            inv_loss = self.c_e_loss(logits, labels)

            valid = valid.reshape(b * t).to(self.device)
            dist_info = DistInfo(prob=F.softmax(logits, dim=1))
            entropy = self.distribution.mean_entropy(
                dist_info=dist_info,
                valid=valid,
            )
            entropy_loss = - self.entropy_loss_coeff * entropy

            correct = torch.argmax(logits.detach(), dim=1) == labels
            accuracy = torch.mean(correct[valid].float())

        else:
            raise NotImplementedError

        perplexity = self.distribution.mean_perplexity(dist_info,
            valid.to(self.device))

        return inv_loss, entropy_loss, accuracy, perplexity, conv_obs

    def validation(self, itr):
        logger.log("Computing validation loss...")
        val_info = ValInfo(*([] for _ in range(len(ValInfo._fields))))
        self.optimizer.zero_grad()
        for _ in range(self.n_validation_batches):
            samples = self.replay_buffer.sample_batch(self.batch_size,
                validation=True)
            with torch.no_grad():
                inv_loss, ent_loss, accuracy, perplexity, conv_output = self.inverse_loss(samples)
            val_info.invLoss.append(inv_loss.item())
            val_info.entLoss.append(ent_loss.item())
            val_info.accuracy.append(accuracy.item())
            val_info.perplexity.append(perplexity.item())
            val_info.convActivation.append(
                conv_output[0].detach().cpu().view(-1).numpy())  # Keep 1 full one.
        self.optimizer.zero_grad()
        logger.log("...validation loss completed.")
        return val_info

    def state_dict(self):
        return dict(
            encoder=self.encoder.state_dict(),
            inverse=self.inverse_model.state_dict(),
            optimizer=self.optimizer.state_dict(),
        )

    def load_state_dict(self, state_dict):
        self.encoder.load_state_dict(state_dict["encoder"])
        self.inverse_model.load_state_dict(state_dict["inverse"])
        self.optimizer.load_state_dict(state_dict["optimizer"])

    def parameters(self):
        yield from self.encoder.parameters()
        yield from self.inverse_model.parameters()

    def named_parameters(self):
        """To allow filtering by name in weight decay."""
        yield from self.encoder.named_parameters()
        yield from self.inverse_model.named_parameters()

    def eval(self):
        self.encoder.eval()  # in case of batch norm
        self.inverse_model.eval()

    def train(self):
        self.encoder.train()
        self.inverse_model.train()
