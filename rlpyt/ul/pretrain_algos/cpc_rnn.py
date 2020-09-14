
import torch
from collections import namedtuple
import pickle
import copy
import torch.nn.functional as F

from rlpyt.utils.tensor import valid_mean
from torch.optim.lr_scheduler import CosineAnnealingLR
from rlpyt.ul.pretrain_algos.utils.warmup_scheduler import GradualWarmupScheduler
from rlpyt.ul.pretrain_algos.utils.weight_decay import add_weight_decay
# from rlpyt.ul.pretrain_algos.data_augs import quick_pad_random_crop
# from rlpyt.utils.tensor import infer_leading_dims

from rlpyt.ul.pretrain_algos.base import UlAlgorithm
from rlpyt.utils.quick_args import save__init__args
from rlpyt.utils.logging import logger
from rlpyt.ul.replays.fixed import FixedReplayFrameBuffer
from rlpyt.utils.buffer import buffer_to
from rlpyt.algos.utils import valid_from_done
from rlpyt.utils.tensor import to_onehot
# from rlpyt.models.utils import update_state_dict
from rlpyt.ul.models.cpc_models import CpcFfEncoderModel

from rlpyt.distributions.categorical import Categorical, DistInfo


IGNORE_INDEX = -100  # Mask CPC samples across episode boundary.
OptInfo = namedtuple("OptInfo", ["cpcLoss",
    "cpcAccuracy1", "cpcAccuracy2", "cpcAccuracyT1", "cpcAccuracyT2", "cpcAccuracy",
    "invLoss", "invEntLoss", "invAccuracy", "invPerplexity",
    "regLoss", "gradNorm",
    "fc1Activation", "zActivation"])
ValInfo = namedtuple("ValInfo", ["cpcLoss",
    "cpcAccuracy1", "cpcAccuracy2", "cpcAccuracyT1", "cpcAccuracyT2", "cpcAccuracy",
    "invLoss", "invEntLoss", "invAccuracy", "invPerplexity",
    "fc1Activation", "zActivation"])


def chain(*iterables):
    for itr in iterables:
        yield from itr


class CpcRnn(UlAlgorithm):

    opt_info_fields = tuple(f for f in OptInfo._fields)  # copy

    def __init__(
            self,
            batch_B,
            batch_T,
            learning_rate,
            replay_filepath,
            contrast_mode="all_delta_t_loop",
            OptimCls=torch.optim.Adam,
            optim_kwargs=None,
            initial_state_dict=None,
            clip_grad_norm=10.,
            validation_split=0.1,
            validation_batch_B=None,
            n_validation_batches=None,
            EncoderCls=CpcFfEncoderModel,
            encoder_kwargs=None,
            ReplayCls=FixedReplayFrameBuffer,
            action_condition=False,
            onehot_actions=True,
            activation_loss_coefficient=0.,  # 0 for OFF
            activation_loss_at="fc1",  # "fc1", "z"
            learning_rate_anneal=None,  # cosine
            learning_rate_warmup=0,  # number of updates
            # data_aug=None,  # [None, "random_crop"]
            # random_crop_pad=4,
            warmup_T=0,
            rnn_size=512,
            bidirectional=False,
            inverse_loss_coeff=0.,
            inverse_ent_coeff=0.
            ):
        optim_kwargs = dict() if optim_kwargs is None else optim_kwargs
        encoder_kwargs = dict() if encoder_kwargs is None else encoder_kwargs
        # transform_kwargs = dict() if transform_kwargs is None else transform_kwargs
        save__init__args(locals())
        self.c_e_loss = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
        assert learning_rate_anneal in [None, "cosine"]
        assert contrast_mode == "all_delta_t_loop"  # only one fast enough
        # assert data_aug in [None, "random_crop"]
        if self.validation_batch_B is None:
            self.validation_batch_B = self.batch_B
        self.batch_size = batch_B * batch_T  # for logging only

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
        # self.target_encoder = copy.deepcopy(self.encoder)
        self.encoder.to(self.device)
        # self.target_encoder.to(self.device)
        
        if not self.action_condition:
            ar_input_size = 0
        else:
            if self.onehot_actions:
                max_act = self.replay_buffer.samples.action.max()
                self._act_dim = max_act + 1  # To use for 1-hot encoding
                ar_input_size = self._act_dim + 1  # for 1 step, + 1 reward
            else:
                assert len(self.replay_buffer.samples.action.shape) == 3
                ar_input_size = self.replay.samples.action.shape[-1] + 1

        
        latent_size = self.encoder_kwargs["latent_size"]
        self.prediction_rnn = torch.nn.LSTM(
            input_size=int(latent_size + ar_input_size),
            hidden_size=self.rnn_size,
            bidirectional=self.bidirectional,
        )
        self.prediction_rnn.to(self.device)
        
        act_dim = self.replay_buffer.samples.action.max() + 1  # [0,1,...,max]
        self.distribution = Categorical(act_dim)
        self.inverse_model = torch.nn.Linear(self.rnn_size, act_dim)
        self.inverse_model.to(self.device)

        transforms = [None] + [
            torch.nn.Linear(in_features=self.rnn_size, out_features=latent_size)
            for _ in range(self.batch_T - 1)]  # no W for delta_t=0
        self.transforms = torch.nn.ModuleList(transforms)
        self.transforms.to(self.device)

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
        # breakpoint()

    def load_replay(self):
        """Right now this loads one replay file...could modify to combine
        multiple, or do that in a script elsewhere."""
        logger.log("Loading replay buffer...")
        with open(self.replay_filepath, "rb") as fh:
            replay_buffer = pickle.load(fh)
        logger.log("Replay buffer loaded")
        self.replay_buffer = self.ReplayCls(
            replay_buffer=replay_buffer,
            sequence_length=self.batch_T,
            validation_split=self.validation_split,
            include_prev_action=True,
        )
        examples = self.replay_buffer.get_examples()
        return examples

    def optimize(self, itr):
        opt_info = OptInfo(*([] for _ in range(len(OptInfo._fields))))
        samples = self.replay_buffer.sample_batch(self.batch_B)
        if self.lr_scheduler is not None:
            self.lr_scheduler.step(itr)  # Do every itr instead of every epoch
        self.optimizer.zero_grad()
        cpc_loss, cpc_accuracies, z_anchor, fc1_anchor, context = self.cpc_loss(samples)
        reg_loss = self.regularization_loss(z_anchor, fc1_anchor)
        inv_loss, inv_ent_loss, inv_accuracy, inv_perplexity = self.inv_loss(samples, context)
        loss = cpc_loss + reg_loss + inv_loss + inv_ent_loss

        loss.backward()
        if self.clip_grad_norm is None:
            grad_norm = 0.
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.parameters(), self.clip_grad_norm)
        self.optimizer.step()
        opt_info.cpcLoss.append(cpc_loss.item())
        opt_info.cpcAccuracy1.append(cpc_accuracies[0].item())
        opt_info.cpcAccuracy2.append(cpc_accuracies[1].item())
        opt_info.cpcAccuracyT1.append(cpc_accuracies[2].item())
        opt_info.cpcAccuracyT2.append(cpc_accuracies[3].item())
        opt_info.cpcAccuracy.append(cpc_accuracies[4].item())
        opt_info.invLoss.append(inv_loss.item())
        opt_info.invEntLoss.append(inv_ent_loss.item())
        opt_info.invAccuracy.append(inv_accuracy.item())
        opt_info.invPerplexity.append(inv_perplexity.item())
        opt_info.regLoss.append(reg_loss.item())
        opt_info.gradNorm.append(grad_norm)
        opt_info.fc1Activation.append(fc1_anchor[0].detach().cpu().numpy())
        opt_info.zActivation.append(z_anchor[0].detach().cpu().numpy())  # Keep 1 full one.
        # if itr % self.target_update_interval == 0:
        #     update_state_dict(self.target_encoder, self.encoder.state_dict(),
        #         self.target_update_tau)
        # Maybe some callback to reduce learning rate or something.
        return opt_info

    def cpc_loss(self, samples):
        observation = samples.observation

        if self.action_condition:
            prev_action = samples.action[:-1]
            if self.onehot_actions:
                prev_action = to_onehot(prev_action,
                    self._act_dim, dtype=torch.float)
            prev_reward = samples.reward[:-1]
        else:
            prev_action = prev_reward = None  # can pass into following functions to ignore
        observation, prev_action, prev_reward = buffer_to(
            (observation, prev_action, prev_reward),
            device=self.device)

        latent, fc1_latent, _ = self.encoder(observation)  # [T,B,..]
        if prev_action is not None:
            rnn_input = torch.cat(
                [latent, prev_action, prev_reward.unsqueeze(-1)],  # [T,B,..]
                dim=-1)
        else:
            rnn_input = latent
        context, _ = self.prediction_rnn(rnn_input)
        if self.bidirectional:
            raise NotImplementedError
            # T, B, H2 = context.shape
            # context = context.view(T, B, 2, H2 / 2)
            # context_backward = context[:, :, 1]
            # context = context[:, :, 0]

        invalid = ~valid_from_done(samples.done).type(torch.bool)
        
        if self.contrast_mode == "B_only":
            loss, accuracies = self._contrast_B_only(latent, context, invalid)
        elif self.contrast_mode == "self_T":
            loss, accuracies = self._contrast_self_T(latent, context, invalid)
        elif self.contrast_mode == "B_and_self_T":
            loss, accuracies = self._contrast_B_and_self_T(latent, context, invalid)
        elif self.contrast_mode == "predictions":
            loss, accuracies = self._contrast_predictions(latent, context, invalid)
        elif self.contrast_mode == "all":
            loss, accuracies = self._contrast_all(latent, context, invalid)
        elif self.contrast_mode == "all_W_loop":
            loss, accuracies = self._contrast_all_W_loop(latent, context, invalid)
        elif self.contrast_mode == "all_delta_t_loop":
            loss, accuracies = self._contrast_all_delta_t_loop(latent, context, invalid)
        else:
            raise NotImplementedError

        return loss, accuracies, latent, fc1_latent, context

    def inv_loss(self, samples, context):
        if self.inverse_loss_coeff == 0.:
            return [torch.tensor(0.)] * 4
        labels = samples.action[1:-1]  # [T-1,B] leave of first PREVIOUS action, and last action
        logits = self.inverse_model(context[1:])  # [T-1,B,A]
        t, b, a = logits.shape
        valid = valid_from_done(samples.done).type(torch.bool)
        valid = valid[1:]  # based on "next" step
        labels[~valid] = IGNORE_INDEX  # based on "next" step
        logits = logits.view(t * b, a)
        labels = labels.view(t * b)
        logits = logits - torch.max(logits, dim=1, keepdim=True)[0]
        loss = self.inverse_loss_coeff * self.c_e_loss(logits, labels)

        valid = valid.view(t * b)
        dist_info = DistInfo(prob=F.softmax(logits, dim=1))
        entropy = self.distribution.mean_entropy(
            dist_info=dist_info,
            valid=valid,
        )
        entropy_loss = - self.inverse_ent_coeff * entropy

        correct = torch.argmax(logits.detach(), dim=1) == labels
        accuracy = torch.mean(correct[valid].float())

        perplexity = self.distribution.mean_perplexity(dist_info, valid)

        return loss, entropy_loss, accuracy, perplexity

    def _contrast_B_only(self, latent, context, invalid):
        T, B, H = latent.shape
        losses = list()
        labels_list = [torch.arange(B, dtype=torch.long, device=self.device)
            for _ in range(T)]
        for t, labels in enumerate(labels_list):
            labels[invalid[t]] = IGNORE_INDEX  # make sure to get this index right
        for t_context in range(self.warmup_T, T - 1):
            for t_latent in range(t_context + 1, T):
                delta_t = t_latent - t_context
                transform = self.transforms[delta_t]  
                prediction = transform(context[t_context])  # need shape [B, H]
                target = latent[t_latent]  # [B, H]
                logits = torch.matmul(prediction, target.T)  # [B, B]
                logits = logits - torch.max(logits, dim=1, keepdim=True)[0]
                # Turn these into loss right now.
                # How to handle invalid?  I think it has to be off the latest time.
                losses.append(self.c_e_loss(logits, labels[t_latent]))
        loss = torch.mean(torch.stack(losses))
        return loss

    def _contrast_self_T(self, latent, context, invalid):
        T, B, H = latent.shape
        losses = list()
        for t_context in range(self.warmup_T, T - 1):
            for t_latent in range(t_context + 1, T):
                delta_t = t_latent - t_context
                transform = self.transforms[delta_t - 1]
                prediction = transform(context[t_context])  # [B, H]
                target = latent[t_context + 1:]  # [t, B, H]
                # label is the same time-step for all in the batch:
                labels = torch.tensor([delta_t - 1] * B, dtype=torch.long, device=self.device)
                labels[invalid[t_latent]] = IGNORE_INDEX
                logits = torch.matmul(
                    prediction.unsqueeze(-1),  # [B, H, 1] 
                    target.transpose(1, 0),  # [B, t, H]
                ).squeeze(-1)  # [B, t, 1] --> [B, t]
                logits = logits - torch.max(logits, dim=1, keepdim=True)[0]
                losses.append(self.c_e_loss(logits, labels))
        loss = torch.mean(torch.stack(losses))
        return loss

    def _contrast_B_and_self_T(self, latent, context, invalid):
        T, B, H = latent.shape
        losses = list()
        labels_list = [torch.arange(B, dtype=torch.long, device=self.device)
            for _ in range(T)]
        for t, labels in enumerate(labels_list):
            labels[invalid[t]] = IGNORE_INDEX  # make sure to get this index right
        for t_context in range(self.warmup_T, T - 1):
            for t_latent in range(t_context + 1, T):
                delta_t = t_latent - t_context
                transform = self.transforms[delta_t]
                prediction = transform(context[t_context])  # [B, H]
                target_B = latent[t_latent]  # [B, H]
                target_T = torch.cat([
                    latent[t_context + 1:t_latent],
                    latent[t_latent + 1:],
                    ], dim=0)  # [t - 1, B, H]  (leave out the positive)
                logits_B = torch.matmul(prediction, target_B.T)  # [B, B]
                logits_T = torch.matmul(
                    prediction.unsqueeze(-1),  # [B, H, 1]
                    target_T.transpose(1, 0),  # [B, t - 1, H]
                ).squeeze(-1)  # [B, t - 1, 1] --> [B, t - 1]
                logits = torch.cat([logits_B, logits_T], dim=-1)  # [B, B + t - 1]
                logits = logits - torch.max(logits, dim=1, keepdim=True)[0]
                losses.append(self.c_e_loss(logits, labels[t_latent]))
        loss = torch.mean(torch.stack(losses))  # Oh, mean or just sum them?
        return loss

    def _contrast_predictions(self, latent, context, invalid):
        T, B, H = latent.shape
        losses = list()
        for t_context in range(self.warmup_T, T - 1):
            for t_latent in range(t_context + 1, T):
                delta_t = t_latent - t_context
                transform = self.transforms[delta_t]
                prediction = transform(context[t_context])  # [B, H]
                target = latent[t_context + 1:]  # [t, B, H]
                # target = target.transpose(1, 0)  # [B, t, H]
                target = target.view(-1, H)  # [t * B, H]
                logits = torch.matmul(
                    prediction,  # [B, H]
                    target.T,  # [H, t * B]
                )  # [B, t * B] # order is: all B at t=0, then all B at t=1, etc.
                logits = logits - torch.max(logits, dim=1, keepdim=True)[0]
                labels = torch.arange(
                    start=(delta_t - 1) * B,
                    end=delta_t * B,
                    step=1,
                    dtype=torch.long,
                    device=self.device,
                )
                labels[invalid[t_latent]] = IGNORE_INDEX
                losses.append(self.c_e_loss(logits, labels))
        loss = torch.mean(torch.stack(losses))  # Oh, mean or just sum them?
        return loss

    def _contrast_all(self, latent, context, invalid):
        T, B, H = latent.shape
        losses = list()
        accuracy1 = list()
        accuracy2 = list()
        accuracy3 = list()
        accuracylast = list()
        accuracyall = list()
        labels_list = [torch.arange(
            start=t * B,
            end=(t + 1) * B,
            step=1,
            dtype=torch.long,
            device=self.device,
            )
            for t in range(T)]
        for t, labels in enumerate(labels_list):
            labels[invalid[t]] = IGNORE_INDEX
        target = latent.view(-1, H)  # [T*B,H]
        for t_context in range(self.warmup_T, T - 1):
            for t_latent in range(t_context + 1, T):
                delta_t = t_latent - t_context
                prediction = self.transforms[delta_t](context[t_context])
                logits = torch.matmul(
                    prediction,  # [B,H]
                    target.T,  # [H,T*B]
                )  # [B,T*B]
                logits = logits - torch.max(logits, dim=1, keepdim=True)[0]
                losses.append(self.c_e_loss(logits, labels_list[t_latent]))
                correct = torch.argmax(logits.detach(), dim=1) == labels_list[t_latent]
                accuracy = torch.mean(correct[~invalid[t_latent]].float())
                accuracyall.append(accuracy)
                if delta_t == 1:
                    accuracy1.append(accuracy)
                elif delta_t == 2:
                    accuracy2.append(accuracy)
                elif delta_t == 3:
                    accuracy3.append(accuracy)
                elif t_context == self.warmup_T and t_latent == T - 1:
                    accuracylast.append(accuracy)
        loss = torch.mean(torch.stack(losses))  # Oh, mean or just sum them?
        accuracies = [torch.mean(torch.stack(acc)) for acc in
            [accuracy1, accuracy2, accuracy3, accuracylast, accuracyall]]
        return loss, accuracies

    def _contrast_all_W_loop(self, latent, context, invalid):
        T, B, H = latent.shape
        target = latent.view(-1, H)  # [T * B, H]
        target_trans = target.T  # [H, T * B]
        context = context.transpose(1, 0)  # [B, T, H]
        # losses = list()
        accuracy1 = list()
        accuracy2 = list()
        accuracyT1 = list()
        accuracyT2 = list()
        labels_list = [torch.arange(
            start=t * B,
            end=(t + 1) * B,
            step=1,
            dtype=torch.long,
            device=self.device,
            )
            for t in range(T)]
        
        for t, labels in enumerate(labels_list):
            labels[invalid[t]] = IGNORE_INDEX  # make sure to get this index right
        
        # Do the all multiplication with Ws in one call for each:
        predictions = [None]
        for delta_t in range(1, T):
            transform = self.transforms[delta_t]
            predictions.append(transform(context[:, :-delta_t]))  # each is [B, t, H]


        # not 100% sure this is right...
        pred_list = list()
        all_labels = list()
        for t_context in range(self.warmup_T, T - 1):
            for t_latent in range(t_context + 1, T):
                delta_t = t_latent - t_context
                prediction = predictions[delta_t][:, t_context]  # [B, H]
                # logits = torch.matmul(
                #     prediction,  # [B, H]
                #     target_trans,  # [H, T * B]
                # )
                # logits = logits - torch.max(logits, dim=1, keepdim=True)[0]
                # losses.append(self.c_e_loss(logits, labels_list[t_latent]))
                pred_list.append(prediction)
                # logits_list.append(logits)
                all_labels.append(labels_list[t_latent])
                
                # accuracyall.append(accuracy)
                # if delta_t == 1 or delta_t == 2 or delta_t == T - 1 or delta_t == T - 2:
                #     correct = torch.argmax(logits.detach(), dim=1) == labels_list[t_latent]
                #     accuracy = torch.mean(correct[~invalid[t_latent]].float())
                #     if delta_t == 1:
                #         accuracy1.append(accuracy)
                #     elif delta_t == 2:
                #         accuracy2.append(accuracy)
                #     elif delta_t == T - 1:
                #         accuracyT1.append(accuracy)
                #     elif delta_t == T - 2:
                #         accuracyT2.append(accuracy)
        pred_all = torch.cat(pred_list)
        logits = torch.matmul(pred_all, target_trans)
        # logits = torch.cat(logits_list)
        logits = logits - torch.max(logits, dim=1, keepdim=True)[0]
        labels = torch.cat(all_labels)
        loss = self.c_e_loss(logits, labels)
        correct = torch.argmax(logits.detach(), dim=1) == labels
        accuracy = valid_mean(correct.float(), valid=labels >= 0)  # IGNORE=-100
        # accuracies = [torch.mean(torch.stack(acc)) for acc in
        #     (accuracy1, accuracy2, accuracyT1, accuracyT2)] + [accuracy]
        accuracies = [torch.tensor(0.)] * 4 + [accuracy]

        # loss = torch.mean(torch.stack(losses))  # Oh, mean or just sum them?
        # accuracies = [torch.mean(torch.stack(acc)) for acc in
        #     [accuracy1, accuracy2, accuracy3, accuracylast, accuracyall]]
        return loss, accuracies
    
    def _contrast_all_delta_t_loop(self, latent, context, invalid):
        T, B, H = latent.shape
        # Should have T,B,C=context.shape; T,B=invalid.shape, all same T,B.
        target_trans = latent.view(-1, H).transpose(1, 0)  # [T,B,H] -> [T*B,H] -> [H,T*B]
        # Draw from base_labels according to the location of the corresponding
        # positive latent for contrast, using [t,b]; will give the location
        # within T*B.
        base_labels = torch.arange(T * B, dtype=torch.long, device=self.device).view(T, B)
        base_labels[invalid] = IGNORE_INDEX  # By location of latent; CELoss ignores these.

        # All predictions and labels into one tensor for efficient contrasting.
        prediction_list = list()
        label_list = list()
        for delta_t in range(1, T - self.warmup_T):
            # Predictions based on context starting from t=0 up to the point where
            # there isn't a future latent within the timesteps of the minibatch.
            # warmup_T is the first time to consider contexts from: let Tw = T-warmup_T
            # (to start thinking about it, imagine warmup_T=0 everywhere, then Tw=T)
            # [Tw-dt,B,C] -> [Tw-dt,B,H] -> [(Tw-dt)*B,H]
            prediction_list.append(self.transforms[delta_t](context[self.warmup_T:-delta_t]).view(-1, H))
            # The correct latent is delta_t time steps ahead:
            # [Tw-dt,B] -> [(Tw-dt)*B]
            label_list.append(base_labels[self.warmup_T + delta_t:].view(-1))

        # Before cat, to isolate delta_t for diagnostic accuracy check later:
        dt_lengths = [0] + [len(label) for label in label_list]
        dtb = torch.cumsum(torch.tensor(dt_lengths), dim=0)  # delta_t_boundaries

        # Total number of predictions: P = Tw*(Tw-1)/2*B
        # from: \sum_{dt=1}^Tw ((Tw-dt) * B)
        predictions = torch.cat(prediction_list)  # [P,H]
        labels = torch.cat(label_list)  # [P]
        # contrast against ALL latents, not just the "future" ones:
        logits = torch.matmul(predictions, target_trans)  # [P,H]*[H,T*B] -> [P,T*B]
        logits = logits - torch.max(logits, dim=1, keepdim=True)[0]  # [P,T*B]
        loss = self.c_e_loss(logits, labels)  # every logit weighted equally

        # Log some downsampled accuracies for diagnostics:
        logits_d = logits.detach()
        # begin, end, step (downsample):
        b, e, s = dtb[0], dtb[1], 4  # delta_t = 1
        logits1, labels1 = logits_d[b:e:s], labels[b:e:s]
        correct1 = torch.argmax(logits1, dim=1) == labels1
        accuracy1 = valid_mean(correct1.float(), valid=labels1 >= 0)  # IGNORE=-100

        b, e, s = dtb[1], dtb[2], 4  # delta_t = 2
        logits2, labels2 = logits_d[b:e:s], labels[b:e:s]
        correct2 = torch.argmax(logits2, dim=1) == labels2
        accuracy2 = valid_mean(correct2.float(), valid=labels2 >= 0)

        b, e, s = dtb[-2], dtb[-1], 1  # delta_t = T - 1
        logitsT1, labelsT1 = logits_d[b:e:s], labels[b:e:s]
        correctT1 = torch.argmax(logitsT1, dim=1) == labelsT1
        accuracyT1 = valid_mean(correctT1.float(), valid=labelsT1 >= 0)

        b, e, s = dtb[-3], dtb[-2], 1  # delta_t = T - 2
        logitsT2, labelsT2 = logits_d[b:e:s], labels[b:e:s]
        correctT2 = torch.argmax(logitsT2, dim=1) == labelsT2
        accuracyT2 = valid_mean(correctT2.float(), valid=labelsT2 >= 0)

        b, e, s = dtb[0], dtb[-1], 8
        logitsAll, labelsAll = logits_d[b:e:s], labels[b:e:s]
        correctAll = torch.argmax(logitsAll, dim=1) == labelsAll
        accuracyAll = valid_mean(correctAll.float(), valid=labelsAll >= 0)

        accuracies = [accuracy1, accuracy2, accuracyT1, accuracyT2, accuracyAll]

        # correct = torch.argmax(logits.detach(), dim=1) == label  # TODO: downsample
        # accuracy = valid_mean(correct.float(), valid=label >= 0)


        # for delta_t in range(1, T):
        #     transform = self.transforms[delta_t]
        #     prediction = transform(context[:-delta_t])  # each is [t, B, H]
        #     prediction = prediction.view(-1, H)  # [t*B, H]
        #     label = labels[delta_t:]  # should be [t, B]
        #     label = label.view(-1)
        #     logit = torch.matmul(prediction, target_trans)
        #     logit = logit - torch.max(logit, dim=1, keepdim=True)[0]
        #     loss = loss + self.c_e_loss(logit, label)
        #     correct = torch.argmax(logit.detach(), dim=1) == label
        #     accuracy = valid_mean(correct.float(), valid=label >= 0)
        # accuracies = [torch.tensor(0.)] * 4 + [accuracy]
        return loss, accuracies
            # to contrast prediction-only, rather than all:
            # target_list.append(latent[delta_t:])  # should be [t, B, H]
        # prediction = torch.cat(prediction_list)  # [tt, B, H]
        # label = torch.cat(label_list)  # [tt, B, H]
        # target = torch.cat(target_list)  # [tt, B, H]
        # target = target.view(-1, H)  # [tt * B, H]
        # logits =

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
            samples = self.replay_buffer.sample_batch(self.validation_batch_B,
                validation=True)
            cpc_loss, cpc_accuracies, z_anchor, fc1_anchor, context = self.cpc_loss(samples)
            inv_loss, inv_ent_loss, inv_accuracy, inv_perplexity = self.inv_loss(samples, context)
            val_info.cpcLoss.append(cpc_loss.item())
            val_info.cpcAccuracy1.append(cpc_accuracies[0].item())
            val_info.cpcAccuracy2.append(cpc_accuracies[1].item())
            val_info.cpcAccuracyT1.append(cpc_accuracies[2].item())
            val_info.cpcAccuracyT2.append(cpc_accuracies[3].item())
            val_info.cpcAccuracy.append(cpc_accuracies[4].item())
            val_info.invLoss.append(inv_loss.item())
            val_info.invEntLoss.append(inv_ent_loss.item())
            val_info.invAccuracy.append(inv_accuracy.item())
            val_info.invPerplexity.append(inv_perplexity.item())
            val_info.fc1Activation.append(fc1_anchor[0].detach().cpu().numpy())
            val_info.zActivation.append(z_anchor[0].detach().cpu().numpy())
        self.optimizer.zero_grad()
        logger.log("...validation loss completed.")
        return val_info

    def state_dict(self):
        return dict(
            encoder=self.encoder.state_dict(),
            # target_encoder=self.target_encoder.state_dict(),
            prediction_rnn=self.prediction_rnn.state_dict(),
            transforms=self.transforms.state_dict(),
            inverse=self.inverse_model.state_dict(),
            optimizer=self.optimizer.state_dict(),
        )

    def load_state_dict(self, state_dict):
        self.encoder.load_state_dict(state_dict["encoder"])
        # self.target_encoder.load_state_dict(state_dict["target_encoder"])
        self.prediction_rnn.load_state_dict(state_dict["prediction_rnn"])
        self.transforms.load_state_dict(state_dict["transforms"])
        self.inverse_model.load_state_dict(state_dict["inverse"])
        self.optimizer.load_state_dict(state_dict["optimizer"])

    def parameters(self):
        yield from self.encoder.parameters()
        yield from self.prediction_rnn.parameters()
        yield from self.transforms.parameters()
        yield from self.inverse_model.parameters()

    def named_parameters(self):
        """To allow filtering by name in weight decay."""
        yield from self.encoder.named_parameters()
        yield from self.prediction_rnn.named_parameters()
        yield from self.transforms.named_parameters()
        yield from self.inverse_model.named_parameters()

    def eval(self):
        self.encoder.eval()  # in case of batch norm
        self.prediction_rnn.eval()
        self.transforms.eval()
        self.inverse_model.eval()

    def train(self):
        self.encoder.train()
        self.prediction_rnn.train()
        self.transforms.train()
        self.inverse_model.train()
