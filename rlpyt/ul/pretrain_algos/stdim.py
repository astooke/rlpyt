
import torch
from collections import namedtuple
import pickle
import copy
import numpy as np

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
from rlpyt.utils.tensor import to_onehot
from rlpyt.models.utils import update_state_dict
from rlpyt.ul.models.stdim_models import StdimEncoderModel, StdimTransformModel


IGNORE_INDEX = -100  # Mask CPC samples across episode boundary.
OptInfo = namedtuple("OptInfo", ["stdim_loss", "gg_loss", "gl_loss", "ll_loss",
    "gg_accuracy", "gl_accuracy", "ll_accuracy",
    "reg_loss", "gradNorm",
    "fc1Activation", "zAbsActivation"])
ValInfo = namedtuple("ValInfo", ["stdim_loss", "gg_loss", "gl_loss", "ll_loss",
    "gg_accuracy", "gl_accuracy", "ll_accuracy",
    "fc1Activation", "zAbsActivation"])


def chain(*iterables):
    for itr in iterables:
        yield from itr


class STDIM(UlAlgorithm):

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
            clip_grad_norm=100.,
            positives="next",
            replay_sequence_length=2,
            use_global_global=False,
            use_global_local=True,
            use_local_local=True,
            local_conv_layer=1,  # 0-based indexing
            validation_split=0.1,
            validation_batch_size=None,
            n_validation_batches=None,
            EncoderCls=StdimEncoderModel,
            encoder_kwargs=None,
            TransformCls=StdimTransformModel,
            transform_kwargs=None,
            ReplayCls=FixedReplayFrameBuffer,
            action_condition=False,
            onehot_actions=True,
            activation_loss_coefficient=0.0,
            activation_loss_at="fc1",  # "fc1", "z"
            learning_rate_anneal=None,  # cosine
            learning_rate_warmup=0,  # number of updates
            data_aug=None,  # [None, "random_crop"]
            random_crop_pad=4,
            sum_losses=False,  # OLD SETTING
            ):
        optim_kwargs = dict() if optim_kwargs is None else optim_kwargs
        encoder_kwargs = dict() if encoder_kwargs is None else encoder_kwargs
        transform_kwargs = dict() if transform_kwargs is None else transform_kwargs
        save__init__args(locals())
        self.c_e_loss = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
        assert learning_rate_anneal in [None, "cosine"]
        assert data_aug in [None, "random_crop"]
        if positives not in ["next", "last"] and "TB_" not in positives:
            raise NotImplementedError
        if self.validation_batch_size is None:
            self.validation_batch_size = self.batch_size

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
            ar_input_size = 0
        else:
            if self.onehot_actions:
                max_act = self.replay_buffer.samples.action.max()
                self._act_dim = max_act + 1  # To use for 1-hot encoding
                ar_input_size = self._act_dim + 1  # for 1 step, + 1 reward
            else:
                assert len(self.replay_buffer.samples.action.shape) == 3  # [T,B,A]
                ar_input_size = self.replay.samples.action.shape[-1] + 1
        #     if self.positives in ["sequence_all", "sequence_next", "last"]:
        #         # Include all actions + rewards, except the last step.
        #         ar_input_size *= self.replay_sequence_length - 1
        # if self.positives == "sequence_all":
        #     transform_length = self.replay_sequence_length  # Maybe?
        # elif self.positives == "sequence_next":
        #     transform_length = self.replay_sequence_length - 1
        # else:
        #     transform_length = 0
        
        global_size = self.encoder_kwargs["latent_size"]
        local_size = self.encoder.conv_out_shapes[self.local_conv_layer][0]
        if self.use_global_global:
            self.gg_transform = self.TransformCls(
                anchor_size=global_size,
                positive_size=global_size,
                # sequence_length=transform_length,
                ar_input_size=ar_input_size,
               **self.transform_kwargs,
            )
            self.gg_transform.to(self.device)
        if self.use_global_local:
            self.gl_transform = self.TransformCls(
                anchor_size=global_size,
                positive_size=local_size,
                # sequence_length=transform_length,
                ar_input_size=ar_input_size,
                **self.transform_kwargs,
            )
            self.gl_transform.to(self.device)
        if self.use_local_local:
            self.ll_transform = self.TransformCls(
                anchor_size=local_size,
                positive_size=local_size,
                # sequence_length=transform_length,
                ar_input_size=ar_input_size,
                **self.transform_kwargs,
            )
            self.ll_transform.to(self.device)

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

        lr_scheduler = None
        if self.learning_rate_anneal == "cosine":
            lr_scheduler = CosineAnnealingLR(self.optimizer,
                T_max=self.n_updates - self.learning_rate_warmup)
        if self.learning_rate_warmup > 0:
            lr_scheduler = GradualWarmupScheduler(
                self.optimizer,
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
        stdim_loss, loss_vals, accuracies, z_anchor, fc1_anchor = self.stdim_loss(samples)
        reg_loss = self.regularization_loss(z_anchor, fc1_anchor)
        loss = stdim_loss + reg_loss
        loss.backward()
        if self.clip_grad_norm is None:
            grad_norm = 0.
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.parameters(), self.clip_grad_norm)
        self.optimizer.step()
        opt_info.stdim_loss.append(stdim_loss.item())
        opt_info.gg_loss.append(loss_vals[0])
        opt_info.gl_loss.append(loss_vals[1])
        opt_info.ll_loss.append(loss_vals[2])
        opt_info.gg_accuracy.append(accuracies[0])
        opt_info.gl_accuracy.append(accuracies[1])
        opt_info.ll_accuracy.append(accuracies[2])
        opt_info.reg_loss.append(reg_loss.item())
        opt_info.gradNorm.append(grad_norm)
        opt_info.fc1Activation.append(fc1_anchor[0].detach().cpu().numpy())
        opt_info.zAbsActivation.append(np.abs(z_anchor[0].detach().cpu().numpy()))  # Keep 1 full one.
        if itr % self.target_update_interval == 0:
            update_state_dict(self.target_encoder, self.encoder.state_dict(),
                self.target_update_tau)
        # Maybe some callback to reduce learning rate or something.
        return opt_info

    def stdim_loss(self, samples):
        if self.replay_sequence_length == 0:
            if self.positives != "anchor":  # or augment?
                raise NotImplementedError
            anchor = positive = samples.observation  # [B,...]
        else:
            anchor = samples.observation[0]  # T=0
            if self.positives == "anchor":
                positive = samples.observation[0]  # same as anchor
            elif self.positives == "next":
                positive = samples.observation[1]  # T=1
            elif self.positives == "last":
                positive = samples.observation[-1]
            elif self.positives == "sequence_all":
                positive = samples.observation  # T >=0
            elif self.positives == "sequence_next":
                positive = samples.observation[1:]  # T >= 1
            elif self.positives.startswith("TB_"):
                delta_T = int(self.positives.lstrip("TB_"))
                anchor = samples.observation[:-delta_T]
                positive = samples.observation[delta_T:]
                t, b, c, h, w = anchor.shape
                anchor = anchor.reshape(t * b, c, h, w)
                positive = positive.reshape(t * b, c, h, w)
            else:
                raise NotImplementedError

        if self.data_aug == "random_crop":
            anchor, rndm_kwargs = quick_pad_random_crop(
                imgs=anchor,
                crop_size=self.image_shape[-1],
                pad=self.random_crop_pad,
            )
            positive, _ = quick_pad_random_crop(
                imgs=positive,  # TODO: decide what to do with sequences
                crop_size=self.image_shape[-1],  # keep it the same, e.g. 84
                pad=self.random_crop_pad,  # just a little, 4?
                **rndm_kwargs
            )

        if self.action_condition:
            assert self.replay_sequence_length > 0
            action = samples.action
            if self.onehot_actions:
                action = to_onehot(action, self._act_dim, dtype=torch.float)
            reward = samples.reward
            if self.positives == "anchor":
                action = reward = None  # no time step
            elif self.positives == "next":
                action = action[:1]  # just the action at T=0, [1,B,A]
                reward = reward[:1]  # same, shape [1,B]
            else:
                action = action[:-1]  # don't need action at last obs
                reward = reward[:-1]  # same
        else:
            action = reward = None  # can pass into following functions to ignore
        anchor, positive, action, reward = buffer_to(
            (anchor, positive, action, reward),
            device=self.device)

        with torch.no_grad():
            z_positive, _, positive_conv_outs = self.target_encoder(positive)
            positive_local = positive_conv_outs[self.local_conv_layer]
        z_anchor, fc1_anchor, anchor_conv_outs = self.encoder(anchor)
        anchor_local = anchor_conv_outs[self.local_conv_layer]

        labels = torch.arange(z_anchor.shape[0],   # batch size
            dtype=torch.long, device=self.device)
        invalid = ~valid_from_done(samples.done).type(torch.bool)
        if self.positives == "next":
            labels[invalid[1]] = IGNORE_INDEX
            nvld = invalid[1]
        elif self.positives == "last":
            labels[invalid[-1]] = IGNORE_INDEX
            nvld = invalid[-1]
        if self.positives.startswith("TB_"):
            delta_T = int(self.positives.lstrip("TB_"))
            nvld = invalid[delta_T:].reshape(-1)  # from same idx as positives
        else:
            raise NotImplementedError

        gg_loss = gl_loss = ll_loss = 0.
        gg_accuracy = gl_accuracy = ll_accuracy = 0.
        if self.use_global_global:
            gg_logits = self.gg_transform(z_anchor, z_positive,
                action, reward)
            gg_loss = self.c_e_loss(gg_logits, labels)
            gg_correct = torch.argmax(gg_logits.detach(), dim=1) == labels
            gg_accuracy = torch.mean(gg_correct[~nvld].float())
        if self.use_global_local:
            gl_logits_list = self.gl_transform(z_anchor, positive_local,
                action, reward)
            gl_losses = torch.stack([self.c_e_loss(gl_logits, labels)
                for gl_logits in gl_logits_list])
            gl_loss = torch.sum(gl_losses) if self.sum_losses else torch.mean(gl_losses)
            gl_corrects = [torch.argmax(gl_logits.detach(), dim=1) == labels
                for gl_logits in gl_logits_list]
            gl_accuracies = [torch.mean(gl_correct[~nvld].float())
                for gl_correct in gl_corrects]
            gl_accuracy = torch.mean(torch.stack(gl_accuracies))
        if self.use_local_local:
            ll_logits_list = self.ll_transform(anchor_local, positive_local,
                action, reward)
            ll_losses = torch.stack([self.c_e_loss(ll_logits, labels)
                for ll_logits in ll_logits_list])
            ll_loss = torch.sum(ll_losses) if self.sum_losses else torch.mean(ll_losses)
            ll_corrects = [torch.argmax(ll_logits.detach(), dim=1) == labels
                for ll_logits in ll_logits_list]
            ll_accuracies = [torch.mean(ll_correct[~nvld].float())
                for ll_correct in ll_corrects]
            ll_accuracy = torch.mean(torch.stack(ll_accuracies))
        stdim_loss = gg_loss + gl_loss + ll_loss
        gg_loss_val = gg_loss if isinstance(gg_loss, float) else gg_loss.item()
        gl_loss_val = gl_loss if isinstance(gl_loss, float) else gl_loss.item()
        ll_loss_val = ll_loss if isinstance(ll_loss, float) else ll_loss.item()
        gg_accuracy = gg_accuracy if isinstance(gg_accuracy, float) else gg_accuracy.item()
        gl_accuracy = gl_accuracy if isinstance(gl_accuracy, float) else gl_accuracy.item()
        ll_accuracy = ll_accuracy if isinstance(ll_accuracy, float) else ll_accuracy.item()

        loss_vals = (gg_loss_val, gl_loss_val, ll_loss_val)
        accuracies = (gg_accuracy, gl_accuracy, ll_accuracy)
        return stdim_loss, loss_vals, accuracies, z_anchor, fc1_anchor

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
            stdim_loss, loss_vals, accuracies, z_anchor, fc1_anchor = self.stdim_loss(samples)
            val_info.stdim_loss.append(stdim_loss.item())
            val_info.gg_loss.append(loss_vals[0])
            val_info.gl_loss.append(loss_vals[1])
            val_info.ll_loss.append(loss_vals[2])
            val_info.gg_accuracy.append(accuracies[0])
            val_info.gl_accuracy.append(accuracies[1])
            val_info.ll_accuracy.append(accuracies[2])
            val_info.fc1Activation.append(fc1_anchor[0].detach().cpu().numpy())
            val_info.zAbsActivation.append(np.abs(z_anchor[0].detach().cpu().numpy()))
        self.optimizer.zero_grad()
        logger.log("...validation loss completed.")
        return val_info

    def state_dict(self):
        state_dict = dict(
            encoder=self.encoder.state_dict(),
            target_encoder=self.target_encoder.state_dict(),
            optimizer=self.optimizer.state_dict(),
        )
        if self.use_global_global:
            state_dict["gg_transform"] = self.gg_transform.state_dict()
        if self.use_global_local:
            state_dict["gl_transform"] = self.gl_transform.state_dict()
        if self.use_local_local:
            state_dict["ll_transform"] = self.ll_transform.state_dict()
        return state_dict

    def load_state_dict(self, state_dict):
        self.encoder.load_state_dict(state_dict["encoder"])
        self.target_encoder.load_state_dict(state_dict["target_encoder"])
        self.optimizer.load_state_dict(state_dict["optimizer"])
        if self.use_global_global:
            self.gg_transform.load_state_dict(state_dict["gg_transform"])
        if self.use_global_local:
            self.gl_transform.load_state_dict(state_dict["gl_transform"])
        if self.use_local_local:
            self.ll_transform.load_state_dict(state_dict["ll_transform"])

    def parameters(self):
        yield from self.encoder.parameters()
        if self.use_global_global:
            yield from self.gg_transform.parameters()
        if self.use_global_local:
            yield from self.gl_transform.parameters()
        if self.use_local_local:
            yield from self.ll_transform.parameters()

    def named_parameters(self):
        """To allow filtering by name in weight decay."""
        yield from self.encoder.named_parameters()
        if self.use_global_global:
            yield from self.gg_transform.named_parameters()
        if self.use_global_local:
            yield from self.gl_transform.named_parameters()
        if self.use_local_local:
            yield from self.ll_transform.named_parameters()

    def eval(self):
        self.encoder.eval()  # in case of batch norm
        if self.use_global_global:
            self.gg_transform.eval()
        if self.use_global_local:
            self.gl_transform.eval()
        if self.use_local_local:
            self.ll_transform.eval()

    def train(self):
        self.encoder.train()
        if self.use_global_global:
            self.gg_transform.train()
        if self.use_global_local:
            self.gl_transform.train()
        if self.use_local_local:
            self.ll_transform.train()
