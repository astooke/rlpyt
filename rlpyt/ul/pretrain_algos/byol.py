
import torch
from collections import namedtuple
import pickle
import copy
import torch.nn.functional as F

from torch.optim.lr_scheduler import CosineAnnealingLR
from rlpyt.ul.pretrain_algos.utils.warmup_scheduler import GradualWarmupScheduler
from rlpyt.ul.pretrain_algos.utils.weight_decay import add_weight_decay
from rlpyt.ul.pretrain_algos.data_augs import quick_pad_random_crop

from rlpyt.ul.pretrain_algos.base import UlAlgorithm
from rlpyt.utils.quick_args import save__init__args
from rlpyt.utils.logging import logger
from rlpyt.ul.replays.fixed import FixedReplayFrameBuffer
from rlpyt.utils.buffer import buffer_to
from rlpyt.algos.utils import valid_from_done
from rlpyt.utils.tensor import to_onehot, valid_mean
from rlpyt.models.utils import update_state_dict
from rlpyt.ul.models.cpc_models import CpcFfEncoderModel
from rlpyt.ul.models.byol_models import ByolFfPredictorModel


IGNORE_INDEX = -100  # Mask CPC samples across episode boundary.
OptInfo = namedtuple("OptInfo", ["byolLoss", "regLoss", "gradNorm",
    "fc1Activation", "zActivation"])
ValInfo = namedtuple("ValInfo", ["byolLoss","fc1Activation", "zActivation"])


def chain(*iterables):
    for itr in iterables:
        yield from itr


class BYOL(UlAlgorithm):

    opt_info_fields = tuple(f for f in OptInfo._fields)  # copy

    def __init__(
            self,
            batch_size,
            learning_rate,
            target_update_tau,   # 1 for hard update
            target_update_interval,
            replay_filepath,
            OptimCls=torch.optim.Adam,
            optim_kwargs=None,
            initial_state_dict=None,
            clip_grad_norm=10.,
            positives="last",
            replay_sequence_length=0,
            validation_split=0.1,
            validation_batch_size=None,
            n_validation_batches=None,
            EncoderCls=CpcFfEncoderModel,
            encoder_kwargs=None,
            PredictorCls=ByolFfPredictorModel,
            predictor_kwargs=None,
            ReplayCls=FixedReplayFrameBuffer,
            action_condition=False,
            onehot_actions=True,
            activation_loss_coefficient=0.01,
            activation_loss_at="fc1",  # "fc1", "z"
            learning_rate_anneal=None,  # cosine
            learning_rate_warmup=0,  # number of updates
            data_aug=None,  # [None, "random_crop"]
            random_crop_pad=4,
            symmetrize=False,
            ):
        optim_kwargs = dict() if optim_kwargs is None else optim_kwargs
        encoder_kwargs = dict() if encoder_kwargs is None else encoder_kwargs
        predictor_kwargs = dict() if predictor_kwargs is None else predictor_kwargs
        save__init__args(locals())
        self.c_e_loss = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
        assert learning_rate_anneal in [None, "cosine"]
        assert data_aug in [None, "random_crop"]
        if validation_batch_size is None:
            self.validation_batch_size = batch_size
        assert not (symmetrize and positives == "sequence")

    def initialize(self, n_updates, cuda_idx=None):
        self.n_updates = n_updates
        self.device = torch.device("cpu") if cuda_idx is None else torch.device(
            "cuda", index=cuda_idx)
        
        examples = self.load_replay()
        if self.n_validation_batches is None:
            self.n_validation_batches = int((self.replay_buffer.size *
                self.validation_split) / self.batch_size)
            logger.log(f"Using {self.n_validation_batches} validation batches.")
        
        self.image_shape = image_shape = examples.observation.shape
        self.encoder = self.EncoderCls(image_shape=image_shape, **self.encoder_kwargs)
        self.target_encoder = copy.deepcopy(self.encoder)
        self.encoder.to(self.device)
        self.target_encoder.to(self.device)
        
        if not self.action_condition:
            seq_input_size = 0
        else:
            if self.onehot_actions:
                max_act = self.replay_buffer.samples.action.max()
                self._act_dim = max_act + 1  # To use for 1-hot encoding
                seq_input_size = self._act_dim + 1  # for 1 step, + 1 reward
            else:
                assert len(self.replay_buffer.samples.action.shape) == 3
                seq_input_size = self.replay.samples.action.shape[-1] + 1
            # Include all actions + rewards, except the last step.
            seq_input_size *= (self.replay_sequence_length - 1)
        if self.positives == "sequence":
            transform_length = self.replay_sequence_length - 1
        else:
            transform_length = 0
        self.predictor = self.PredictorCls(
            latent_size=self.encoder_kwargs["latent_size"],
            sequence_length=transform_length,
            seq_input_size=seq_input_size,
            **self.predictor_kwargs
        )
        self.predictor.to(self.device)
        if self.symmetrize:
            self.reverse_predictor = copy.deepcopy(self.predictor)

        weight_decay = self.optim_kwargs.pop("weight_decay", 0.)
        parameters, weight_decay = add_weight_decay(
            model=self,  # has .parameters() and .named_parameters()
            weight_decay=weight_decay,
            filter_ndim_1=True,
            skip_list=None,
        )
        self.optimizer = self.OptimCls(
            parameters,
            lr=self.learning_rate,
            weight_decay=weight_decay,
            **self.optim_kwargs
        )
        # breakpoint()

        lr_scheduler = None
        if self.learning_rate_anneal == "cosine":
            lr_scheduler = CosineAnnealingLR(self.optimizer,
                T_max=self.n_updates - self.learning_rate_warmup)
        if self.learning_rate_warmup > 0:
            lr_scheduler = GradualWarmupScheduler(self.optimizer,
                multiplier=1,
                total_epoch=self.learning_rate_warmup,  # actually n_updates
                after_scheduler=lr_scheduler,
        )
        self.lr_scheduler = lr_scheduler
        if lr_scheduler is not None:
            self.optimizer.zero_grad()
            self.optimizer.step()  # possibly needed to initialize the scheduler?

        if self.initial_state_dict is not None:
            self.load_state_dict(self.initial_state_dict)

    def load_replay(self):
        """Right now this loads one replay file...could modify to combine
        multiple, or do that in a script elsewhere."""
        logger.log("Loading replay buffer...")
        with open(self.replay_filepath, "rb") as fh:
            replay_buffer = pickle.load(fh)
        logger.log("Replay buffer loaded")
        self.replay_buffer = self.ReplayCls(
            replay_buffer=replay_buffer,
            sequence_length=self.replay_sequence_length,
            validation_split=self.validation_split,
        )
        examples = self.replay_buffer.get_examples()
        return examples

    def optimize(self, itr):
        opt_info = OptInfo(*([] for _ in range(len(OptInfo._fields))))
        samples = self.replay_buffer.sample_batch(self.batch_size)
        if self.lr_scheduler is not None:
            self.lr_scheduler.step(itr)  # Do every itr instead of every epoch
        self.optimizer.zero_grad()
        byol_loss, z_anchor, fc1_anchor = self.byol_loss(samples)
        reg_loss = self.regularization_loss(z_anchor, fc1_anchor)
        loss = byol_loss + reg_loss
        loss.backward()
        if self.clip_grad_norm is None:
            grad_norm = 0.
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.parameters(), self.clip_grad_norm)
        self.optimizer.step()
        opt_info.byolLoss.append(byol_loss.item())
        opt_info.regLoss.append(reg_loss.item())
        opt_info.gradNorm.append(grad_norm)
        opt_info.fc1Activation.append(fc1_anchor[0].detach().cpu().numpy())
        opt_info.zActivation.append(z_anchor[0].detach().cpu().numpy())  # Keep 1 full one.
        if itr % self.target_update_interval == 0:
            update_state_dict(self.target_encoder, self.encoder.state_dict(),
                self.target_update_tau)
        # Maybe some callback to reduce learning rate or something.
        return opt_info

    def byol_loss(self, samples):
        anchor = samples.observation[0]
        if self.positives == "last":
            positive = samples.observation[-1]
        elif self.positives == "sequence":
            positive = samples.observation[1:]
        else:
            raise NotImplementedError

        if self.action_condition:
            assert self.replay_sequence_length > 0
            action = samples.action
            if self.onehot_actions:
                action = to_onehot(action, self._act_dim, dtype=torch.float)
            reward = samples.reward
            action = action[:-1]  # don't need action at last obs
            reward = reward[:-1]  # same
        else:
            action = reward = None  # can pass into following functions to ignore
        anchor, positive, action, reward = buffer_to(
            (anchor, positive, action, reward),
            device=self.device)

        with torch.no_grad():
            z_positive, _, _ = self.target_encoder(positive)
        z_anchor, fc1_anchor, _ = self.encoder(anchor)
        q_anchor = self.predictor(z_anchor, action, reward)

        losses = self._byol_losses(q_anchor, z_positive)

        valid = valid_from_done(samples.done)
        if self.positives == "last":
            valid = valid[-1]
        elif self.positives == "sequence":
            valid = valid[1:]
        valid = valid.to(self.device)

        loss = valid_mean(losses, valid)

        if self.symmetrize:
            with torch.no_grad():
                z_sym_positive, _, _ = self.target_encoder(anchor)
            z_sym_anchor, _, _ = self.encoder(positive)
            q_sym = self.reverse_predictor(z_sym_anchor, action, reward)
            sym_losses = self._byol_losses(q_sym, z_sym_positive)
            sym_loss = valid_mean(sym_losses, valid)
            loss = loss + sym_loss

        return loss, z_anchor, fc1_anchor

    def _byol_losses(self, q, z):
        q = F.normalize(q, dim=-1, p=2)
        z = F.normalize(z, dim=-1, p=2)
        losses = 2. - 2 * (q * z).sum(dim=-1)
        return losses

    def regularization_loss(self, z_anchor, fc1_anchor):
        if self.activation_loss_coefficient == 0.:
            return torch.tensor(0.)
        if self.activation_loss_at == "z":
            # L2 norm but mean over latent instead of sum, to count elementwise.
            norm_z = torch.sqrt(z_anchor.pow(2).mean(dim=1))
            # MSE loss to try to keep the avg close to 1?
            # Might unecessarily keep things on one side or other of 0.
            reg_loss = 0.5 * (norm_z - 1).pow(2).mean()
        elif self.activation_loss_at == "fc1":
            # Only penalize above 1
            # (abs here should be redundant, fc1_anchor after relu)
            large_x = torch.clamp(torch.abs(fc1_anchor) - 1, min=0.)
            
            # NO: sqrt was throwing nan, anyway, we instead want each neuron?
            # # L2-style loss on the large activations of the vector.
            # large_x_l2_sq = torch.sqrt(large_x.pow(2).sum(dim=-1))  # could fix with +1e-6
            # reg_loss = large_x_l2_sq.mean()  # Average over batch.

            # Gentle squared-magnitude loss, l2-like
            reg_loss = large_x.pow(2).mean()
        else:
            raise NotImplementedError
        return self.activation_loss_coefficient * reg_loss

    def validation(self, itr):
        logger.log("Computing validation loss...")
        val_info = ValInfo(*([] for _ in range(len(ValInfo._fields))))
        self.optimizer.zero_grad()
        for _ in range(self.n_validation_batches):
            samples = self.replay_buffer.sample_batch(self.validation_batch_size,
                validation=True)
            byol_loss, z_anchor, fc1_anchor = self.byol_loss(samples)
            val_info.byolLoss.append(byol_loss.item())
            val_info.fc1Activation.append(fc1_anchor[0].detach().cpu().numpy())
            val_info.zActivation.append(z_anchor[0].detach().cpu().numpy())
        self.optimizer.zero_grad()
        logger.log("...validation loss completed.")
        return val_info

    def state_dict(self):
        state_dict = dict(
            encoder=self.encoder.state_dict(),
            target_encoder=self.target_encoder.state_dict(),
            predictor=self.predictor.state_dict(),
            optimizer=self.optimizer.state_dict(),
        )
        if self.symmetrize:
            state_dict["reverse_predictor"] = self.reverse_predictor.state_dict()
        return state_dict

    def load_state_dict(self, state_dict):
        self.encoder.load_state_dict(state_dict["encoder"])
        self.target_encoder.load_state_dict(state_dict["target_encoder"])
        self.predictor.load_state_dict(state_dict["predictor"])
        self.optimizer.load_state_dict(state_dict["optimizer"])
        if self.symmetrize:
            self.reverse_predictor.load_state_dict(state_dict["reverse_predictor"])

    def parameters(self):
        yield from self.encoder.parameters()
        yield from self.predictor.parameters()
        if self.symmetrize:
            yield from self.reverse_predictor.parameters()

    def named_parameters(self):
        """To allow filtering by name in weight decay."""
        yield from self.encoder.named_parameters()
        yield from self.predictor.named_parameters()
        if self.symmetrize:
            yield from self.reverse_predictor.named_parameters()

    def eval(self):
        self.encoder.eval()  # in case of batch norm
        self.predictor.eval()
        if self.symmetrize:
            self.reverse_predictor.eval()

    def train(self):
        self.encoder.train()
        self.predictor.train()
        if self.symmetrize:
            self.reverse_predictor.train()
