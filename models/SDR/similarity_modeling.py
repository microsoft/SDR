from argparse import ArgumentParser
import logging
import math
from pytorch_metric_learning.distances.lp_distance import LpDistance

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.nn.functional import gelu
from pytorch_metric_learning import miners, losses, reducers

from transformers.configuration_roberta import RobertaConfig
from transformers.modeling_bert import BertLayerNorm, BertPreTrainedModel
from transformers.modeling_roberta import RobertaModel, RobertaLMHead
from pytorch_metric_learning.distances import CosineSimilarity


class SimilarityModeling(BertPreTrainedModel):
    config_class = RobertaConfig
    base_model_prefix = "roberta"

    def __init__(self, config, hparams):
        super().__init__(config)
        self.hparams = hparams
        config.output_hidden_states = True

        self.roberta = RobertaModel(config)
        self.lm_head = RobertaLMHead(config)
        self.init_weights()

        if self.hparams.metric_for_similarity == "cosine":
            self.metric = CosineSimilarity()
            pos_margin, neg_margin = 1, 0
            neg_margin = -1 if (getattr(self.hparams, "metric_loss_func", "ContrastiveLoss") == "CosineLoss") else 0
        elif self.hparams.metric_for_similarity == "norm_euc":
            self.metric = LpDistance(normalize_embeddings=True, p=2)
            pos_margin, neg_margin = 0, 1

        self.reducer = reducers.DoNothingReducer()
        if self.hparams.hard_mine:
            self.miner_func = miners.MultiSimilarityMiner()
        else:
            self.miner_func = miners.BatchEasyHardMiner(
                pos_strategy=miners.BatchEasyHardMiner.ALL,
                neg_strategy=miners.BatchEasyHardMiner.ALL,
                distance=CosineSimilarity(),
            )

        if getattr(self.hparams, "metric_loss_func", "ContrastiveLoss") in ["ContrastiveLoss", "CosineLoss"]:
            self.similarity_loss_func = losses.ContrastiveLoss(
                pos_margin=pos_margin, neg_margin=neg_margin, distance=self.metric
            )  # |np-sp|_+ + |sn-mn|_+ so for cossim we do pos_m=1 and neg_m=0
        else:
            self.similarity_loss_func = losses.TripletMarginLoss(margin=1, distance=self.metric)

    def get_output_embeddings(self):
        return self.lm_head.decoder

    @staticmethod
    def mean_mask(features, mask):
        return (features * mask.unsqueeze(-1)).sum(1) / mask.sum(-1, keepdim=True)

    def forward(
        self,
        input_ids=None,
        sample_labels=None,
        samples_idxs=None,
        track_sim_dict=None,
        non_masked_input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        masked_lm_labels=None,
        labels=None,
        output_hidden_states=False,
        return_dict=False,
        run_similarity=False,
        run_mlm=True,
    ):
        if run_mlm:
            outputs = list(
                self.roberta(
                    input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    position_ids=position_ids,
                    head_mask=head_mask,
                    inputs_embeds=inputs_embeds,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )
            )
            sequence_output = outputs[0]
            prediction_scores = self.lm_head(sequence_output)
            outputs = (prediction_scores, None, sequence_output)  # Add hidden states and attention if they are here

            #######
            # MLM
            #######
            masked_lm_loss = torch.zeros(1, device=prediction_scores.device).float()

            if (masked_lm_labels is not None and (not (masked_lm_labels == -100).all())) and self.hparams.mlm:
                loss_fct = CrossEntropyLoss()
                masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            else:
                masked_lm_loss = 0
        else:
            outputs = (
                torch.zeros([*input_ids.shape, 50265]).to(input_ids.device).float(),
                None,
                torch.zeros([*input_ids.shape, 1024]).to(input_ids.device).float(),
            )
            masked_lm_loss = torch.zeros(1)[0].to(input_ids.device).float()

        #######
        # Similarity
        #######
        if run_similarity:
            non_masked_outputs = self.roberta(
                non_masked_input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            non_masked_seq_out = non_masked_outputs[0]

            meaned_sentences = non_masked_seq_out.mean(1)
            miner_output = list(self.miner_func(meaned_sentences, sample_labels))

            sim_loss = self.similarity_loss_func(meaned_sentences, sample_labels, miner_output)
            outputs = (masked_lm_loss, sim_loss, torch.zeros(1)) + outputs
        else:
            outputs = (
                masked_lm_loss,
                torch.zeros(1)[0].to(input_ids.device).float(),
                torch.zeros(1)[0].to(input_ids.device).float(),
            ) + outputs

        return outputs
