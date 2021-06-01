from data.datasets import (
    WikipediaTextDatasetParagraphsSentences,
    WikipediaTextDatasetParagraphsSentencesTest,
)
from utils.argparse_init import str2bool
from models.SDR.SDR_utils import MPerClassSamplerDeter
from data.data_utils import get_gt_seeds_titles, reco_sentence_collate, reco_sentence_test_collate
from functools import partial
import os
from models.reco.hierarchical_reco import vectorize_reco_hierarchical
from utils.torch_utils import to_numpy
from models.transformers_base import TransformersBase
from models.doc_similarity_pl_template import DocEmbeddingTemplate
from utils import switch_functions
from models import transformer_utils
import numpy as np
import torch
from utils import metrics_utils
from pytorch_metric_learning.samplers import MPerClassSampler
from torch.utils.data.dataloader import DataLoader
import json


class SDR(TransformersBase):

    """
    Author: Dvir Ginzburg.

    SDR model (ACL IJCNLP 2021)
    """

    def __init__(
        self, hparams,
    ):
        """Stub."""
        super(SDR, self).__init__(hparams)


    def forward_train(self, batch):
        inputs, labels = transformer_utils.mask_tokens(batch[0].clone().detach(), self.tokenizer, self.hparams)

        outputs = self.model(
            inputs,
            masked_lm_labels=labels,
            non_masked_input_ids=batch[0],
            sample_labels=batch[-1],
            run_similarity=True,
            run_mlm=True,
        )

        self.losses["mlm_loss"] = outputs[0]
        self.losses["d2v_loss"] = (outputs[1] or 0)  * self.hparams.sim_loss_lambda # If no similarity loss we ignore

        tracked = self.track_metrics(input_ids=inputs, outputs=outputs, is_train=self.hparams.mode == "train", labels=labels,)
        self.tracks.update(tracked)

        return

    def forward_val(self, batch):
        self.forward_train(batch)

    def test_step(self, batch, batch_idx):
        section_out = []
        for section in batch[0]:  # batch=1 for test
            sentences=[]
            sentences_embed_per_token = [
                self.model(
                    sentence.unsqueeze(0), masked_lm_labels=None, run_similarity=False
                )[5].squeeze(0)
                for sentence in section[0][:8]
            ]
            for idx, sentence in enumerate(sentences_embed_per_token):
                sentences.append(sentence[: section[2][idx]].mean(0))  # We take the non-padded tokens and mean them
            section_out.append(torch.stack(sentences))
        return (section_out, batch[0][0][1][0])  # title name

    def forward(self, batch):
        eval(f"self.forward_{self.hparams.mode}")(batch)

    @staticmethod
    def track_metrics(
        outputs=None, input_ids=None, labels=None, is_train=True, batch_idx=0,
    ):
        mode = "train" if is_train else "val"

        trackes = {}
        lm_pred = np.argmax(outputs[3].cpu().detach().numpy(), axis=2)
        labels_numpy = labels.cpu().numpy()
        labels_non_zero = labels_numpy[np.array(labels_numpy != -100)] if np.any(labels_numpy != -100) else np.zeros(1)
        lm_pred_non_zero = lm_pred[np.array(labels_numpy != -100)] if np.any(labels_numpy != -100) else np.ones(1)
        lm_acc = torch.tensor(
            metrics_utils.simple_accuracy(lm_pred_non_zero, labels_non_zero), device=outputs[3].device,
        ).reshape((1, -1))

        trackes["lm_acc_{}".format(mode)] = lm_acc.detach().cpu()

        return trackes

    def test_epoch_end(self, outputs, recos_path=None):
        if self.trainer.checkpoint_callback.last_model_path == "" and self.hparams.resume_from_checkpoint is None:
            self.trainer.checkpoint_callback.last_model_path = f"{self.hparams.hparams_dir}/no_train"
        elif(self.hparams.resume_from_checkpoint is not None):
            self.trainer.checkpoint_callback.last_model_path = self.hparams.resume_from_checkpoint
        if recos_path is None:
            save_outputs_path = f"{self.trainer.checkpoint_callback.last_model_path}_FEATURES_NumSamples_{len(outputs)}"

            if isinstance(outputs[0][0][0], torch.Tensor):
                outputs = [([to_numpy(section) for section in sample[0]], sample[1]) for sample in outputs]
            torch.save(outputs, save_outputs_path)
            print(f"\nSaved to {save_outputs_path}\n")

            titles = popular_titles = [out[1][:-1] for out in outputs]
            idxs, gt_path = list(range(len(titles))), ""

            section_sentences_features = [out[0] for out in outputs]
            popular_titles, idxs, gt_path = get_gt_seeds_titles(titles, self.hparams.dataset_name)

            self.hparams.test_sample_size = (
                self.hparams.test_sample_size if self.hparams.test_sample_size > 0 else len(popular_titles)
            )
            idxs = idxs[: self.hparams.test_sample_size]

            recos, metrics = vectorize_reco_hierarchical(
                all_features=section_sentences_features,
                titles=titles,
                gt_path=gt_path,
                output_path=self.trainer.checkpoint_callback.last_model_path,
            )
            metrics = {
                "mrr": float(metrics["mrr"]),
                "mpr": float(metrics["mpr"]),
                **{f"hit_rate_{rate[0]}": float(rate[1]) for rate in metrics["hit_rates"]},
            }
            print(json.dumps(metrics, indent=2))
            for k, v in metrics.items():
                self.logger.experiment.add_scalar(k, v, global_step=self.global_step)

            recos_path = os.path.join(
                os.path.dirname(self.trainer.checkpoint_callback.last_model_path),
                f"{os.path.basename(self.trainer.checkpoint_callback.last_model_path)[:-5]}"
                f"_numSamples_{self.hparams.test_sample_size}",
            )
            torch.save(recos, recos_path)
            print("Saving recos in {}".format(recos_path))

            setattr(self.hparams, "recos_path", recos_path)
        return

    def dataloader(self, mode=None):
        if mode == "train":
            sampler = MPerClassSampler(
                self.train_dataset.labels,
                2,
                batch_size=self.hparams.train_batch_size,
                length_before_new_iter=(self.hparams.limit_train_batches) * self.hparams.train_batch_size,
            )

            loader = DataLoader(
                self.train_dataset,
                num_workers=self.hparams.num_data_workers,
                sampler=sampler,
                batch_size=self.hparams.train_batch_size,
                collate_fn=partial(reco_sentence_collate, tokenizer=self.tokenizer,),
            )

        elif mode == "val":
            sampler = MPerClassSamplerDeter(
                self.val_dataset.labels,
                2,
                length_before_new_iter=self.hparams.limit_val_indices_batches,
                batch_size=self.hparams.val_batch_size,
            )

            loader = DataLoader(
                self.val_dataset,
                num_workers=self.hparams.num_data_workers,
                sampler=sampler,
                batch_size=self.hparams.val_batch_size,
                collate_fn=partial(reco_sentence_collate, tokenizer=self.tokenizer,),
            )

        else:
            loader = DataLoader(
                self.test_dataset,
                num_workers=self.hparams.num_data_workers,
                batch_size=self.hparams.test_batch_size,
                collate_fn=partial(reco_sentence_test_collate, tokenizer=self.tokenizer,),
            )
        return loader

    @staticmethod
    def add_model_specific_args(parent_parser, task_name, dataset_name, is_lowest_leaf=False):
        parser = TransformersBase.add_model_specific_args(parent_parser, task_name, dataset_name, is_lowest_leaf=False)
        parser.add_argument("--hard_mine", type=str2bool, nargs="?", const=True, default=True)
        parser.add_argument("--metric_loss_func", type=str, default="ContrastiveLoss")  # TripletMarginLoss #CosineLoss
        parser.add_argument("--sim_loss_lambda", type=float, default=0.1)
        parser.add_argument("--limit_tokens", type=int, default=64)
        parser.add_argument("--limit_val_indices_batches", type=int, default=500)
        parser.add_argument("--metric_for_similarity", type=str, choices=["cosine", "norm_euc"], default="cosine")

        parser.set_defaults(
            check_val_every_n_epoch=1,
            batch_size=32,
            accumulate_grad_batches=2,
            metric_to_track="train_mlm_loss_epoch",
        )

        return parser

    def prepare_data(self):
        block_size = (
            self.hparams.block_size
            if hasattr(self.hparams, "block_size")
            and self.hparams.block_size > 0
            and self.hparams.block_size < self.tokenizer.max_len
            else self.tokenizer.max_len
        )
        self.train_dataset = WikipediaTextDatasetParagraphsSentences(
            tokenizer=self.tokenizer,
            hparams=self.hparams,
            dataset_name=self.hparams.dataset_name,
            block_size=block_size,
            mode="train",
        )
        self.val_dataset = WikipediaTextDatasetParagraphsSentences(
            tokenizer=self.tokenizer,
            hparams=self.hparams,
            dataset_name=self.hparams.dataset_name,
            block_size=block_size,
            mode="val",
        )
        self.val_dataset.indices_map = self.val_dataset.indices_map[: self.hparams.limit_val_indices_batches]
        self.val_dataset.labels = self.val_dataset.labels[: self.hparams.limit_val_indices_batches]

        self.test_dataset = WikipediaTextDatasetParagraphsSentencesTest(
            tokenizer=self.tokenizer,
            hparams=self.hparams,
            dataset_name=self.hparams.dataset_name,
            block_size=block_size,
            mode="test",
        )

