import pickle
import numpy as np
from tqdm import tqdm
from pathlib import Path
import sys
from utils.logging_utils import Unbuffered
import json


def evaluate_wiki_recos(recos, output_path, gt_path, examples):
    original_stdout = sys.stdout
    sys.stdout = Unbuffered(open(f"{output_path}_reco_scores", "w"))
    dataset_as_dict = {sample[1]: sample for sample in examples}
    recos_as_dict = {reco[0]: reco for reco in recos}
    names_to_id = {sample[1]: idx for idx, sample in enumerate(examples)}
    titles = [ex[1] for ex in examples]

    article_recos_per_article = pickle.load(open(gt_path, "rb"))

    percentiles, mpr = calculate_mpr(recos_as_dict, article_recos_per_article, dataset_as_dict, names_to_id, titles=titles)
    recepricals, mrr = calculate_mrr(recos_as_dict, article_recos_per_article, dataset_as_dict, names_to_id, titles=titles)
    hit_rates, hit_rate = calculate_mean_hit_rate(
        recos_as_dict,
        article_recos_per_article,
        dataset_as_dict,
        names_to_id,
        rate_thresulds=[5, 10, 50, 100, 1000],
        titles=titles,
    )

    metrics = {"mrr": float(mrr), "mpr": float(mpr), **{f"hit_rate_{rate[0]}": float(rate[1]) for rate in hit_rate}}
    print(json.dumps(metrics, indent=2))

    sys.stdout = original_stdout
    return percentiles, mpr, recepricals, mrr, hit_rates, hit_rate


def calculate_mpr(
    input_recommandations, article_article_gt, dataset, names_to_id, sample_size=-1, popular_titles=None, titles=[]
):
    percentiles = []
    for reco_idx in tqdm(input_recommandations):
        wiki_title = titles[reco_idx]
        curr_gts, text = [], []
        recommandations = input_recommandations[reco_idx][1]
        if wiki_title not in article_article_gt:
            continue
        for gt_title in article_article_gt[wiki_title].keys():
            lookup = gt_title.replace("&", "&amp;") if "amp;" not in gt_title and gt_title not in names_to_id else gt_title
            if lookup not in names_to_id:
                print(f"{lookup} not in names_to_id")
                continue
            recommended_idx_ls = np.where(recommandations == names_to_id[lookup])[0]
            if recommended_idx_ls.shape[0] == 0:
                continue
            curr_gts.append(recommended_idx_ls[0])
            percentiles.extend((recommended_idx_ls[0] / len(recommandations),) * article_article_gt[wiki_title][gt_title])
            text.append("gt: {}    gt place: {}".format(gt_title, recommended_idx_ls[0]))

        if len(curr_gts) > 0:
            print(
                "title: {}\n".format(wiki_title)
                + "\n".join(text)
                + "\ntopk: {}\n\n\n".format([titles[reco_i] for reco_i in recommandations[:10]])
            )

    percentiles = percentiles if percentiles != [] else [0]
    print("percentiles_mean:{}\n\n\n\n".format(sum(percentiles) / len(percentiles)))
    return percentiles, sum(percentiles) / len(percentiles)


def calculate_mrr(
    input_recommandations, article_article_gt, dataset, names_to_id, sample_size=-1, popular_titles=None, titles=[]
):
    """
        input_recommandations - list of [] per title the order of all titles recommended with it
        article_article_gt - dict of dicts, each element is a sample, and all the gt samples goes with it and the count each sample
        sample_size - the amount of candidates to calculate the MPR on
    """
    recepricals = []
    for reco_idx in tqdm(input_recommandations):
        wiki_title = titles[reco_idx]
        text = []
        recommandations = input_recommandations[reco_idx][1]
        top = len(input_recommandations)
        for gt_title in article_article_gt[wiki_title].keys():
            lookup = gt_title.replace("&", "&amp;") if "amp;" not in gt_title and gt_title not in names_to_id else gt_title
            if lookup not in names_to_id:
                print(f"{lookup} not in names_to_id")
                continue
            recommended_idx_ls = np.where(recommandations == names_to_id[lookup])[0]
            if recommended_idx_ls.shape[0] > 0 and recommended_idx_ls[0] < top:
                top = recommended_idx_ls[0]
            if recommended_idx_ls.shape[0] == 0:
                continue
            text.append("gt: {}    gt place: {} ".format(gt_title, recommended_idx_ls[0]))

        if top == 0:
            top = 1

        if len(text) > 0:
            recepricals.append(1 / (top))
            text.append(f"\n receprical: {recepricals[-1]}")
            print(
                "title: {}\n".format(wiki_title)
                + "\n".join(text)
                + "\ntopk: {}\n\n\n".format([titles[reco_i] for reco_i in recommandations[:10]])
            )

    recepricals = recepricals if recepricals != [] else [0]
    print(f"Recepricle mean:{sum(recepricals) / len(recepricals)}")
    print(f"Recepricals \n {recepricals}")
    return recepricals, sum(recepricals) / len(recepricals)


def calculate_mean_hit_rate(
    input_recommandations,
    article_article_gt,
    dataset,
    names_to_id,
    sample_size=-1,
    popular_titles=None,
    rate_thresulds=[100],
    titles=[],
):
    mean_hits = [[] for i in rate_thresulds]
    for reco_idx in tqdm(input_recommandations):
        wiki_title = titles[reco_idx]
        curr_gts, text = [], []
        hit_by_rate = [0 for i in rate_thresulds]
        recommandations = input_recommandations[reco_idx][1]
        for gt_title in article_article_gt[wiki_title].keys():

            lookup = gt_title.replace("&", "&amp;") if "amp;" not in gt_title and gt_title not in names_to_id else gt_title
            if lookup not in names_to_id:
                print(f"{lookup} not in names_to_id")
                continue
            recommended_idx_ls = np.where(recommandations == names_to_id[lookup])[0]
            for thr_idx, thresuld in enumerate(rate_thresulds):
                if recommended_idx_ls.shape[0] != 0 and recommended_idx_ls[0] < thresuld:
                    hit_by_rate[thr_idx] += 1
            text.append(f"gt: {gt_title}    gt place: {recommended_idx_ls}")

        if len(text) > 0:
            for thr_idx, thresuld in enumerate(rate_thresulds):
                print(
                    f"title: {wiki_title} Hit rate at {thresuld}: {hit_by_rate[thr_idx]} \n \n {''.join(text)} \n topk: {[titles[reco_i] for reco_i in recommandations[:10]]}\n\n\n"
                )
                hit_mean = hit_by_rate[thr_idx] / len(article_article_gt[wiki_title].keys()) if hit_by_rate[thr_idx] > 0 else 0
                mean_hits[thr_idx].append(hit_mean)

    mean_hits = mean_hits if mean_hits != [[] for rate_thresuld in rate_thresulds] else [[0] for rate_thresuld in rate_thresulds]
    mean_hit = [sum(mean_hit) / len(mean_hit) for mean_hit in mean_hits]
    mean_hits_with_thresuld = [(thresuld, mean) for (thresuld, mean) in zip(*[rate_thresulds, mean_hit])]
    print(f"Hit rate mean:{mean_hits_with_thresuld}")
    return mean_hits, mean_hits_with_thresuld
