# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass, field

import torch
from torch import nn
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from omegaconf import II
import random
import logging

class TemporaryGrad(object):
    def __enter__(self):
        self.prev = torch.is_grad_enabled()
        torch.set_grad_enabled(True)
    def __exit__(self, exc_type: any, exc_value: any, traceback: any) -> None:
        torch.set_grad_enabled(self.prev)

@dataclass
class LabelSmoothedCrossEntropyCriterionConfig(FairseqDataclass):
    label_smoothing: float = field(
        default=0.0,
        metadata={"help": "epsilon for label smoothing, 0 means no label smoothing"},
    )
    report_accuracy: bool = field(
        default=False,
        metadata={"help": "report accuracy metric"},
    )
    ignore_prefix_size: int = field(
        default=0,
        metadata={"help": "Ignore first N tokens"},
    )
    sentence_avg: bool = II("optimization.sentence_avg")
    weight_var_loss: float = field(
        default=0.5,
        metadata={"help": "weight of variance penalty loss"},
    )
    weight_reg_loss: float = field(
        default=0.2,
        metadata={"help": "weight of variance penalty loss"},
    )
    weight_D: float = field(
        default=0.1,
        metadata={"help": "weight of discriminator loss"},
    )
    weight_G: float = field(
        default=0.1,
        metadata={"help": "weight of generator loss"},
    )


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / (lprobs.size(-1) - 1)
    loss = (1.0 - epsilon - eps_i) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


@register_criterion(
    "label_smoothed_cross_entropy", dataclass=LabelSmoothedCrossEntropyCriterionConfig
)
class LabelSmoothedCrossEntropyCriterion(FairseqCriterion):
    def __init__(
        self,
        task,
        sentence_avg,
        label_smoothing,
        ignore_prefix_size=0,
        report_accuracy=False,
        weight_var_loss=0.5,
        weight_reg_loss=0.2,
        weight_D=0.1,
        weight_G=0.1,
    ):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.eps = label_smoothing
        self.ignore_prefix_size = ignore_prefix_size
        self.report_accuracy = report_accuracy
        self.weight_var_loss = weight_var_loss
        self.weight_reg_loss = weight_reg_loss
        self.weight_D = weight_D
        self.weight_G = weight_G
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
        self.bce_loss = nn.BCELoss(reduction='none')
        self.mse_loss = nn.MSELoss(reduction='none')

    def jensen_shannon_mie(self, pos, neg):
        assert pos.shape == neg.shape, (pos.shape, neg.shape)
        pos = - torch.log(1 + torch.exp(-pos))
        neg = torch.log(1 + torch.exp(neg))
        mi = pos - neg
        return mi

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output, aux_outputs = model(**sample["net_input"])
        loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": loss.data,
            "loss_asr": loss.data,
            "nll_loss": nll_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)

        ### MI Discriminator
        Discriminator = model.encoder.w2v_model.mi_discriminator

        ############## Discriminator
        features_video, features_audio = aux_outputs['feats']  # (B, T, D)
        ## a. feats mi
        features_video_d = features_video.clone().detach()
        features_audio_d = features_audio.clone().detach()

        B, T, D = features_video_d.size()
        pos_logits = Discriminator(features_video_d, features_audio_d)              # (B, T, 1)

        shuf_features_audio_d = [f for f in features_audio_d.flatten(0, 1)]
        random.shuffle(shuf_features_audio_d)
        shuf_features_audio_d = torch.stack(shuf_features_audio_d, dim=0).view(B, T, -1)    # (B, T, D)
        neg_logits = Discriminator(features_video_d, shuf_features_audio_d)         # (B, T, 1)

        loss_feats_mi = self.jensen_shannon_mie(pos_logits, neg_logits)
        mask = torch.isfinite(loss_feats_mi) & ~torch.isnan(loss_feats_mi)
        loss_feats_mi = loss_feats_mi.masked_select(mask).sum() / B
        logging_output["loss_feats_mi"] = loss_feats_mi.data

        ## b. centers mi
        vcenters, pcenters = aux_outputs['centers']  # (N, D)
        if vcenters is not None and pcenters is not None:
            vcenters_d = vcenters.clone().detach()
            pcenters_d = pcenters.clone().detach()

            N, D = vcenters_d.size()
            pos_logits_d = Discriminator(vcenters_d, pcenters_d)            # (N, 1)

            shuf_pcenters_d = [c for c in pcenters_d]
            random.shuffle(shuf_pcenters_d)
            shuf_pcenters_d = torch.stack(shuf_pcenters_d, dim=0)           # [N, D]
            neg_logits_d = Discriminator(vcenters_d, shuf_pcenters_d)       # [N, 1]

            loss_centers_mi_d = self.jensen_shannon_mie(pos_logits_d, neg_logits_d)
            mask_d = torch.isfinite(loss_centers_mi_d) & ~torch.isnan(loss_centers_mi_d)
            new_B = N / T
            loss_centers_mi_d = loss_centers_mi_d.masked_select(mask_d).sum() / new_B
            logging_output["loss_centers_mi_d"] = loss_centers_mi_d.data
        else:
            loss_centers_mi_d = torch.Tensor([0]).to(features_video.device)
            logging_output["loss_centers_mi_d"] = loss_centers_mi_d.data

        ## c. v2p_transfer mi
        v2p_transfer = aux_outputs['v2p_transfer']                  # (B, T, D)
        if v2p_transfer is not None:
            v2p_transfer_d = v2p_transfer.clone().detach()
            features_video_d2 = features_video.clone().detach()
            B, T, D = v2p_transfer_d.size()
            assert features_video_d2.size() == (B, T, D)
            pos_logits_d = Discriminator(features_video_d2, v2p_transfer_d)  # (B, T, 1)

            v2p_transfer_shuffle_d = [f for f in v2p_transfer_d.flatten(0, 1)]
            random.shuffle(v2p_transfer_shuffle_d)
            v2p_transfer_shuffle_d = torch.stack(v2p_transfer_shuffle_d, dim=0).view(B, T, -1)
            neg_logits_d = Discriminator(features_video_d2, v2p_transfer_shuffle_d)  # (B, T, 1)

            loss_v2p_mi_d = self.jensen_shannon_mie(pos_logits_d, neg_logits_d)
            mask_d = torch.isfinite(loss_v2p_mi_d) & ~torch.isnan(loss_v2p_mi_d)
            loss_v2p_mi_d = loss_v2p_mi_d.masked_select(mask_d).sum() / B
            logging_output["loss_v2p_mi_d"] = loss_v2p_mi_d.data
        else:
            loss_v2p_mi_d = torch.Tensor([0]).to(features_video.device)
            logging_output["loss_v2p_mi_d"] = loss_v2p_mi_d.data

        ## 1. Discriminator loss =  - feats mi + (centers mi & v2p_transfer mi)
        loss_D = (- loss_feats_mi + loss_centers_mi_d + loss_v2p_mi_d) * self.weight_D
        logging_output["loss_D"] = loss_D.data

        ############## Generator
        ## d. centers mi
        if vcenters is not None and pcenters is not None:
            N, D = vcenters.size()
            pos_logits = Discriminator(vcenters, pcenters)  # (N, 1)

            shuf_pcenters = [c for c in pcenters]
            random.shuffle(shuf_pcenters)
            shuf_pcenters = torch.stack(shuf_pcenters, dim=0)  # [N, D]
            neg_logits = Discriminator(vcenters, shuf_pcenters)  # [N, 1]

            loss_centers_mi = self.jensen_shannon_mie(pos_logits, neg_logits)
            mask = torch.isfinite(loss_centers_mi) & ~torch.isnan(loss_centers_mi)
            new_B = N / T
            loss_centers_mi = loss_centers_mi.masked_select(mask).sum() / new_B
            logging_output["loss_centers_mi"] = loss_centers_mi.data
        else:
            loss_centers_mi = torch.Tensor([0]).to(features_video.device)
            logging_output["loss_centers_mi"] = loss_centers_mi.data

        ## e. v2p_transfer mi
        v2p_transfer = aux_outputs['v2p_transfer']  # (B, T, D)
        if v2p_transfer is not None:
            B, T, D = v2p_transfer.size()
            assert features_video.size() == (B, T, D)
            pos_logits = Discriminator(features_video, v2p_transfer)    # (B, T, 1)

            v2p_transfer_shuffle = [f for f in v2p_transfer.flatten(0, 1)]
            random.shuffle(v2p_transfer_shuffle)
            v2p_transfer_shuffle = torch.stack(v2p_transfer_shuffle, dim=0).view(B, T, -1)
            neg_logits = Discriminator(features_video, v2p_transfer_shuffle)    # (B, T, 1)

            loss_v2p_mi = self.jensen_shannon_mie(pos_logits, neg_logits)
            mask = torch.isfinite(loss_v2p_mi) & ~torch.isnan(loss_v2p_mi)
            loss_v2p_mi = loss_v2p_mi.masked_select(mask).sum() / B
            logging_output["loss_v2p_mi"] = loss_v2p_mi.data

            loss_regression = self.mse_loss(v2p_transfer, features_audio)
            mask = torch.isfinite(loss_regression) & ~torch.isnan(loss_regression)
            loss_regression = loss_regression.masked_select(mask).sum() * self.weight_reg_loss / B
            logging_output["loss_regression"] = loss_regression.data
        else:
            loss_v2p_mi = torch.Tensor([0]).to(features_video.device)
            logging_output["loss_v2p_mi"] = loss_v2p_mi.data

            loss_regression = torch.Tensor([0]).to(features_video.device) * self.weight_reg_loss
            logging_output["loss_regression"] = loss_regression.data

        ## 2. Generator loss = - (centers mi & v2p_transfer mi)
        loss_G = (- loss_centers_mi - loss_v2p_mi) * self.weight_G
        logging_output["loss_G"] = loss_G.data

        ## variance loss
        loss_var =  - aux_outputs['var_penalty'].to(features_video.device) * self.weight_var_loss
        logging_output["loss_var"] = loss_var.data

        loss = loss + loss_G + loss_regression + loss_var
        logging_output["loss"] = loss.data

        return loss_D, loss, sample_size, logging_output


    def get_lprobs_and_target(self, model, net_output, sample):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        target = model.get_targets(sample, net_output)
        if self.ignore_prefix_size > 0:
            if getattr(lprobs, "batch_first", False):
                lprobs = lprobs[:, self.ignore_prefix_size :, :].contiguous()
                target = target[:, self.ignore_prefix_size :].contiguous()
            else:
                lprobs = lprobs[self.ignore_prefix_size :, :, :].contiguous()
                target = target[self.ignore_prefix_size :, :].contiguous()
        return lprobs.view(-1, lprobs.size(-1)), target.view(-1)

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs,
            target,
            self.eps,
            ignore_index=self.padding_idx,
            reduce=reduce,
        )
        return loss, nll_loss

    def compute_accuracy(self, model, net_output, sample):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        mask = target.ne(self.padding_idx)
        n_correct = torch.sum(
            lprobs.argmax(1).masked_select(mask).eq(target.masked_select(mask))
        )
        total = torch.sum(mask)
        return n_correct, total

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        loss_asr_sum = sum(log.get("loss_asr", 0) for log in logging_outputs)
        loss_feats_mi_sum = sum(log.get("loss_feats_mi", 0) for log in logging_outputs)
        loss_centers_mi_sum = sum(log.get("loss_centers_mi", 0) for log in logging_outputs)
        loss_v2p_mi_sum = sum(log.get("loss_v2p_mi", 0) for log in logging_outputs)
        loss_regression_sum = sum(log.get("loss_regression", 0) for log in logging_outputs)
        loss_var_sum = sum(log.get("loss_var", 0) for log in logging_outputs)

        nll_loss_sum = sum(log.get("nll_loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "loss_asr", loss_asr_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "loss_feats_mi", loss_feats_mi_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "loss_centers_mi", loss_centers_mi_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "loss_v2p_mi", loss_v2p_mi_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "loss_regression", loss_regression_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "loss_var", loss_var_sum / sample_size / math.log(2), sample_size, round=3
        )

        metrics.log_scalar(
            "nll_loss", nll_loss_sum / ntokens / math.log(2), ntokens, round=3
        )
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
        )

        total = utils.item(sum(log.get("total", 0) for log in logging_outputs))
        if total > 0:
            metrics.log_scalar("total", total)
            n_correct = utils.item(
                sum(log.get("n_correct", 0) for log in logging_outputs)
            )
            metrics.log_scalar("n_correct", n_correct)
            metrics.log_derived(
                "accuracy",
                lambda meters: round(
                    meters["n_correct"].sum * 100.0 / meters["total"].sum, 3
                )
                if meters["total"].sum > 0
                else float("nan"),
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
