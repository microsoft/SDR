import json
from data.data_utils import get_gt_seeds_titles
from models.reco.wiki_recos_eval.eval_metrics import evaluate_wiki_recos
from utils.torch_utils import mean_non_pad_value, to_numpy
from models.reco.recos_utils import index_amp, sim_matrix
import numpy as np
import torch
from tqdm import tqdm
import pickle
from sklearn.preprocessing import normalize


def vectorize_reco_hierarchical(all_features, titles,gt_path, output_path=""):
    gt = pickle.load(open(gt_path, "rb"))
    to_reco_indices = [index_amp(titles, title) for title in gt.keys()]
    to_reco_indices = list(filter(lambda title: title != None, to_reco_indices))
    sections_per_article = np.array([len(article) for article in all_features])
    sections_per_article_cumsum = np.array([0,] + [len(article) for article in all_features]).cumsum()
    features_per_section = [sec for article in all_features for sec in article]
    features_per_section_torch = [torch.from_numpy(feat) for feat in features_per_section]
    features_per_section_padded = torch.nn.utils.rnn.pad_sequence(
        features_per_section_torch, batch_first=True, padding_value=torch.tensor(float("nan"))
    ).cuda()

    num_samples, max_after_pad = features_per_section_padded.shape[:2]

    flattened = features_per_section_padded.reshape(-1, features_per_section_padded.shape[-1])

    recos = []
    for i in tqdm(to_reco_indices):
        if i > len(all_features):
            print(f"GT title {titles[i]} was not evaluated")
            continue

        to_reco_flattened = features_per_section_padded[
            sections_per_article_cumsum[i] : sections_per_article_cumsum[i + 1]
        ].reshape(-1, features_per_section_padded.shape[-1])

        sim = sim_matrix(to_reco_flattened, flattened)
        reshaped_sim = sim.reshape(
            sections_per_article_cumsum[i + 1] - sections_per_article_cumsum[i], max_after_pad, num_samples, max_after_pad
        )
        sim = reshaped_sim.permute(0, 2, 1, 3)
        sim[sim.isnan()] = float("-Inf")
        score_mat = sim.max(-1)[0]
        score = mean_non_pad_value(score_mat, axis=-1, pad_value=torch.tensor(float("-Inf")).cuda())

        score_per_article = torch.split(score.t(), sections_per_article.tolist(), dim=0)
        score_per_article_padded = torch.nn.utils.rnn.pad_sequence(
            score_per_article, batch_first=True, padding_value=float("-Inf")
        ).permute(0, 2, 1)
        score_per_article_padded[torch.isnan(score_per_article_padded)] = float("-Inf")
        par_score_mat = score_per_article_padded.max(-1)[0]
        par_score = mean_non_pad_value(par_score_mat, axis=-1, pad_value=float("-Inf"))

        recos.append((i, to_numpy(par_score.argsort(descending=True)[1:])))

    examples = [[None, title] for title in titles]  # reco_utils compatibale
    _, mpr, _, mrr, _, hit_rate = evaluate_wiki_recos(recos, output_path, gt_path, examples=examples)
    metrics = {"mrr": mrr, "mpr": mpr, "hit_rates": hit_rate}
    return recos, metrics

