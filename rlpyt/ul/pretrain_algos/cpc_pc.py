
import torch
from collections import namedtuple
import pickle
import copy

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
from rlpyt.utils.tensor import to_onehot, select_at_indexes
from rlpyt.models.utils import update_state_dict
from rlpyt.ul.models.cpc_models import CpcFfEncoderModel, CpcFfTransformModel
from rlpyt.ul.models.pixel_control_models import PixelControlModel

IGNORE_INDEX = -100  # Mask CPC samples across episode boundary.
OptInfo = namedtuple("OptInfo", ["cpcLoss", "accuracy", "pixctlLoss", "regLoss",
    "gradNorm", "fc1Activation", "zActivation"])
ValInfo = namedtuple("ValInfo", ["cpcLoss", "accuracy", "pixctlLoss",
    "fc1Activation", "zActivation"])


def chain(*iterables):
    for itr in iterables:
        yield from itr


class CpcPc(UlAlgorithm):

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
            positives="anchor",
            replay_sequence_length=0,
            validation_split=0.1,
            validation_batch_size=None,
            n_validation_batches=None,
            EncoderCls=CpcFfEncoderModel,
            encoder_kwargs=None,
            TransformCls=CpcFfTransformModel,
            transform_kwargs=None,
            ReplayCls=FixedReplayFrameBuffer,
            action_condition=False,
            onehot_actions=True,
            activation_loss_coefficient=0.01,
            activation_loss_at="fc1",  # "fc1", "z"
            learning_rate_anneal=None,  # cosine
            learning_rate_warmup=0,  # number of updates
            data_aug=None,  # [None, "random_crop"]
            random_crop_pad=4,
            random_crop_prob=1.,
            downsample_rate_T=1.,
            downsample_start_B=None,
            downsample_stop_B=None,
            downsample_step_B=1,
            intensity_sigma=0.,
            intensity_prob=1.,
            B_sample_mode="mixed",  # "single", "two"
            pixel_control=True,
            pixel_control_coeff=1.,
            PixCtlModelCls=PixelControlModel,
            pixel_control_replay_kwargs=None,
            pixel_control_model_kwargs=None,
            pixel_control_filename="pixel_control_80x80_4x4.pkl",
            cpc_loss_coeff=1.,
            # pc_min_c_new=-1,
            # pc_max_c_new=None,
            ):
        optim_kwargs = dict() if optim_kwargs is None else optim_kwargs
        encoder_kwargs = dict() if encoder_kwargs is None else encoder_kwargs
        transform_kwargs = dict() if transform_kwargs is None else transform_kwargs
        pixel_control_replay_kwargs = dict() if pixel_control_replay_kwargs is None else pixel_control_replay_kwargs
        pixel_control_model_kwargs = dict() if pixel_control_model_kwargs is None else pixel_control_model_kwargs
        save__init__args(locals())
        self.c_e_loss = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
        assert learning_rate_anneal in [None, "cosine"]
        assert data_aug in [None, "random_crop", "random_crop_same"]
        if validation_batch_size is None:
            self.validation_batch_size = batch_size

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
        
        if self.onehot_actions:
            max_act = self.replay_buffer.samples.action.max()
            self._act_dim = max_act + 1  # To use for 1-hot encoding
        else:
            # Need to make pixel control with action as INPUT to Q-network.
            raise NotImplementedError
        if not self.action_condition:
            seq_input_size = 0
        else:
            if self.onehot_actions:
                seq_input_size = self._act_dim + 1  # for 1 step, + 1 reward
            else:
                assert len(self.replay_buffer.samples.action.shape) == 3
                seq_input_size = self.replay.samples.action.shape[-1] + 1
            if self.positives in ["sequence_all", "sequence_next", "last"]:
                # Include all actions + rewards, except the last step.
                seq_input_size *= self.replay_sequence_length - 1
        if self.positives == "sequence_all":
            transform_length = self.replay_sequence_length  # Maybe?
        elif self.positives == "sequence_next":
            transform_length = self.replay_sequence_length - 1
        else:
            transform_length = 0
        self.transform = self.TransformCls(
            latent_size=self.encoder_kwargs["latent_size"],
            sequence_length=transform_length,
            seq_input_size=seq_input_size,
            **self.transform_kwargs
        )
        self.transform.to(self.device)

        if self.pixel_control:
            self.pixel_control_input = self.pixel_control_model_kwargs.pop("input", "fc1")
            if self.pixel_control_input == "fc1":
                pc_input_shape = self.encoder_kwargs["fc1_size"]
            elif self.pixel_control_input == "z":
                pc_input_shape = self.encoder_kwargs["latent_size"]
            elif self.pixel_control_input == "conv":
                pc_input_shape = self.encoder.conv_out_shape
            elif self.pixel_control_input == "samples":
                pc_input_shape = self.encoder.conv_out_shape
            channels = self.pixel_control_model_kwargs.pop("channels", [])
            channels.append(self._act_dim)
            self.pixel_control_model = self.PixCtlModelCls(
                input_shape=pc_input_shape,
                channels=channels,
                **self.pixel_control_model_kwargs
            )
        self.pixel_control_model.to(self.device)

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
        """This loads either one or multiple replay buffer files."""
        if isinstance(self.replay_filepath, (list, tuple)):
            logger.log("Loading multiple replay buffers...")
            replay_buffers = list()
            pixel_control_values = list()
            for rep_file in self.replay_filepath:
                with open(rep_file, "rb") as fh:
                    replay_buffers.append(pickle.load(fh))
                pc_file = rep_file.rsplit("/", 1)[0] + "/" + self.pixel_control_filename
                with open(pc_file, "rb") as fh:
                    pixel_control_values.append(pickle.load(fh))
            logger.log("Replay buffers loaded; combining...")
            self.replay_buffer = self.ReplayCls(
                replay_buffers=replay_buffers,
                sequence_length=self.replay_sequence_length,
                validation_split=self.validation_split,
                downsample_rate_T=self.downsample_rate_T,
                downsample_start_B=self.downsample_start_B,
                downsample_stop_B=self.downsample_stop_B,
                downsample_step_B=self.downsample_step_B,
                B_sample_mode=self.B_sample_mode,
                pixel_control_values=pixel_control_values,
            )
            logger.log("Replay buffers combined")
        else:
            logger.log("Loading replay buffer...")
            with open(self.replay_filepath, "rb") as fh:
                replay_buffer = pickle.load(fh)
            pc_file = self.replay_filepath.rsplit("/", 1)[0] + "/" + self.pixel_control_filename
            with open(pc_file, "rb") as fh:
                pixel_control_values = pickle.load(fh)
            self.replay_buffer = self.ReplayCls(
                replay_buffer=replay_buffer,
                sequence_length=self.replay_sequence_length,
                validation_split=self.validation_split,
                downsample_rate_T=self.downsample_rate_T,
                downsample_start_B=self.downsample_start_B,
                downsample_stop_B=self.downsample_stop_B,
                downsample_step_B=self.downsample_step_B,
                B_sample_mode=self.B_sample_mode,
                pixel_control_values=pixel_control_values,
            )
        examples = self.replay_buffer.get_examples()
        return examples

    def optimize(self, itr):
        opt_info = OptInfo(*([] for _ in range(len(OptInfo._fields))))
        samples = self.replay_buffer.sample_batch(self.batch_size)
        if self.lr_scheduler is not None:
            self.lr_scheduler.step(itr)  # Do every itr instead of every epoch
        self.optimizer.zero_grad()
        if self.cpc_loss_coeff > 0.:
            cpc_loss, accuracy, z_anchor, fc1_anchor, conv_anchor = self.cpc_loss(samples)
        else:
            cpc_loss = torch.tensor(0.)
            accuracy = torch.tensor(0.)
            z_anchor = fc1_anchor = conv_anchor = None
        pc_loss = self.pixel_control_loss(samples, z_anchor, fc1_anchor, conv_anchor)
        reg_loss = self.regularization_loss(z_anchor, fc1_anchor)
        if self.cpc_loss_coeff > 0.:
            loss = cpc_loss + pc_loss + reg_loss
        else:
            loss = pc_loss + reg_loss  # was getting a cuda vs cpu backprop error?
        loss.backward()
        if self.clip_grad_norm is None:
            grad_norm = 0.
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.parameters(), self.clip_grad_norm)
        self.optimizer.step()
        opt_info.cpcLoss.append(cpc_loss.item())
        opt_info.accuracy.append(accuracy.item())
        opt_info.pixctlLoss.append(pc_loss.item())
        opt_info.regLoss.append(reg_loss.item())
        opt_info.gradNorm.append(grad_norm)
        if fc1_anchor is not None:
            opt_info.fc1Activation.append(fc1_anchor[0].detach().cpu().numpy())
        if z_anchor is not None:
            opt_info.zActivation.append(z_anchor[0].detach().cpu().numpy())  # Keep 1 full one.
        if itr % self.target_update_interval == 0:
            update_state_dict(self.target_encoder, self.encoder.state_dict(),
                self.target_update_tau)
        # Maybe some callback to reduce learning rate or something.
        return opt_info

    def cpc_loss(self, samples):
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

        if self.data_aug in ["random_crop", "random_crop_same"]:
            anchor, rndm_kwargs = quick_pad_random_crop(
                imgs=anchor,
                # crop_size=self.image_shape[-1],  # Let the function do it.
                pad=self.random_crop_pad,
                prob=self.random_crop_prob,
            )
            if self.data_aug == "random_crop":
                rndm_kwargs = dict()  # don't use the same random values
            positive, _ = quick_pad_random_crop(
                imgs=positive,  # TODO: decide what to do with sequences
                # crop_size=self.image_shape[-1],  # keep it the same, e.g. 84
                pad=self.random_crop_pad,  # just a little, 4?
                prob=self.random_crop_prob,
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
            z_positive, _, _ = self.target_encoder(positive)
        z_anchor, fc1_anchor, conv_anchor = self.encoder(anchor)
        logits = self.transform(z_anchor, z_positive, action, reward)

        labels = torch.arange(z_anchor.shape[0],
            dtype=torch.long, device=self.device)
        invalid = ~valid_from_done(samples.done).type(torch.bool)

        
        if self.positives == "sequence_all":
            loss = self.c_e_loss(logits[0], labels)
            correct = torch.argmax(logits[0].detach(), dim=1) == labels
            accuracy = [torch.mean(correct.float())]

            for lgts, nvld in zip(logits[1:], invalid[1:]):
                labels = labels.clone()
                labels[nvld] = IGNORE_INDEX
                loss += self.c_e_loss(lgts, labels)
                correct = torch.argmax(lgts.detach(), dim=1) == labels
                accuracy.append(torch.mean(correct[~nvld].float()))
            accuracy = torch.mean(torch.stack(accuracy))
        elif self.positives == "sequence_next":
            loss = 0.
            accuracy = []
            for lgts, nvld in zip(logits, invalid[1:]):
                labels = labels.clone()
                labels[nvld] = IGNORE_INDEX
                loss += self.c_e_loss(lgts, labels)
                correct = torch.argmax(lgts.detach(), dim=1) == labels
                accuracy.append(torch.mean(correct[~nvld].float()))
            accuracy = torch.mean(torch.stack(accuracy))

        else:
            if self.positives == "next":
                nvld = invalid[1]
            if self.positives == "last":
                nvld = invalid[-1]
            if self.positives.startswith("TB_"):
                delta_T = int(self.positives.lstrip("TB_"))
                nvld = invalid[delta_T:].reshape(-1)  # from same idx as positives
            labels[nvld] = IGNORE_INDEX
            loss = self.c_e_loss(logits, labels)
            correct = torch.argmax(logits.detach(), dim=1) == labels
            accuracy = torch.mean(correct[~nvld].float())

        return loss, accuracy, z_anchor, fc1_anchor, conv_anchor

    def pixel_control_loss(self, samples, z_anchor, fc1_anchor, conv_anchor):
        # breakpoint()
        if not self.pixel_control or self.pixel_control_coeff == 0.:
            return torch.tensor(0.)
        if self.data_aug is not None and self.random_crop_prob > 0.:
            # compute the encoder on non-shifted inputs
            if self.positives.startswith("TB_"):
                delta_T = int(self.positives.lstrip("TB_"))
                anchor = samples.observation[:-delta_T]
                t, b, c, h, w = anchor.shape
                anchor = anchor.reshape(t * b, c, h, w)
            else:
                anchor = samples.observation[0]
            z_anchor, fc1_anchor, conv_anchor = self.encoder(anchor)
        if self.pixel_control_input == "z":
            pc_input = z_anchor
        elif self.pixel_control_input == "fc1":
            pc_input = fc1_anchor
        elif self.pixel_control_input == "conv":
            pc_input = conv_anchor
        if self.pixel_control_input == "samples":
            if self.positives.startswith("TB_"):
                delta_T = int(self.positives.lstrip("TB_"))
                obs = samples.observation[:-delta_T]
                t, b, c, h, w = obs.shape
                obs = obs.reshape(t * b, c, h, w)
            elif self.positives == "last":
                obs = samples.observation[0]
            else:
                raise NotImplementedError
            obs = buffer_to(obs, device=self.device)
            _, _, pc_input = self.encoder(obs)  # take the conv
        B = pc_input.shape[0]  # no leading T dim [B,..]
        pc_return = samples.pixctl_return
        action = samples.action
        if len(pc_return.shape) == 4:  # assume replay gave sequence [T,B,H',W']
            if self.positives.startswith("TB_"):
                delta_T = int(self.positives.lstrip("TB_"))
                pc_return = pc_return[:-delta_T]
                t, b, h, w = pc_return.shape
                pc_return = pc_return.reshape(t * b, h, w)  # [T,B,H',W'] -> [T*B,H',W']
                action = action[:-delta_T].reshape(-1)  # [T,B] -> [B]
            elif self.positives == "last":
                pc_return = pc_return[0]  # [T,B,H',W'] -> [B,H',W']
                action = action[0]  # [T,B] -> [B]
            else:
                raise NotImplementedError
        pc_input, action, pc_return = buffer_to(
            (pc_input, action, pc_return), device=self.device)
        q_pc = self.pixel_control_model(pc_input)  # [B,A,H',W']
        q_pc_at_a = q_pc[torch.arange(B), action]  # [B,H',W']
        pc_losses = 0.5 * (q_pc_at_a - pc_return) ** 2  # [B,H',W']
        pc_losses = pc_losses.sum(dim=(1, 2))  # [B]  SUM across cells
        pc_loss = self.pixel_control_coeff * pc_losses.mean()  # MEAN across batch
        return pc_loss

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
            if self.cpc_loss_coeff > 0.:
                cpc_loss, accuracy, z_anchor, fc1_anchor, conv_anchor = self.cpc_loss(samples)
            else:
                cpc_loss = torch.tensor(0.)
                accuracy = torch.tensor(0.)
                z_anchor = fc1_anchor = conv_anchor = None
            pc_loss = self.pixel_control_loss(samples, z_anchor, fc1_anchor, conv_anchor)
            val_info.cpcLoss.append(cpc_loss.item())
            val_info.accuracy.append(accuracy.item())
            val_info.pixctlLoss.append(pc_loss.item())
            if fc1_anchor is not None:
                val_info.fc1Activation.append(fc1_anchor[0].detach().cpu().numpy())
            if z_anchor is not None:
                val_info.zActivation.append(z_anchor[0].detach().cpu().numpy())
        self.optimizer.zero_grad()
        logger.log("...validation loss completed.")
        return val_info

    def state_dict(self):
        state_dict = dict(
            encoder=self.encoder.state_dict(),
            target_encoder=self.target_encoder.state_dict(),
            transform=self.transform.state_dict(),
            optimizer=self.optimizer.state_dict(),
        )
        if self.pixel_control:
            state_dict["pixel_control"] = self.pixel_control_model.state_dict()
        return state_dict

    def load_state_dict(self, state_dict):
        self.encoder.load_state_dict(state_dict["encoder"])
        self.target_encoder.load_state_dict(state_dict["target_encoder"])
        self.transform.load_state_dict(state_dict["transform"])
        self.optimizer.load_state_dict(state_dict["optimizer"])
        if self.pixel_control:
            self.pixel_control_model.load_state_dict(state_dict["pixel_control"])

    def parameters(self):
        yield from self.encoder.parameters()
        yield from self.transform.parameters()
        if self.pixel_control:
            yield from self.pixel_control_model.parameters()

    def named_parameters(self):
        """To allow filtering by name in weight decay."""
        yield from self.encoder.named_parameters()
        yield from self.transform.named_parameters()
        if self.pixel_control:
            yield from self.pixel_control_model.named_parameters()

    def eval(self):
        self.encoder.eval()  # in case of batch norm
        self.transform.eval()
        if self.pixel_control:
            self.pixel_control_model.eval()

    def train(self):
        self.encoder.train()
        self.transform.train()
        if self.pixel_control:
            self.pixel_control_model.train()
