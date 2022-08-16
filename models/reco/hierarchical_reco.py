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


def vectorize_reco_hierarchical(all_features, titles,gt_path,config, output_path=""):
    gt = pickle.load(open(gt_path, "rb"))
    
    if not config.allTitles:
        cached_features_file = '/home/jonathanE/Desktop/Github/SDR/data/datasets/cached_proccessed/video_games2/bs_512_video_games_WikipediaTextDatasetParagraphsSentencesTest_tokenizer_RobertaTokenizer_mode_test'
        with open(cached_features_file, "rb") as handle:
            examplesName, indices_map = pickle.load(handle)
        titleSet = hasGamePlaySection(examplesName,config.titleFilterByTopicName)

    else:
        titleSet = set([it for it in titles if len(it)>0 ])


    if not config.filterSummaryByIndex==None and config.allTitles:
        all_features = [[feat[config.filterSummaryByIndex]] if len(feat)>config.filterSummaryByIndex else [] for feat in all_features]
    all_features2 = [all_features[ind] for ind in np.arange(len(all_features)) if len(all_features[ind])>0 and titles[ind] in titleSet]
    titles = [it for it in titles if len(it)>0]

    if config.numberOfSentences==None:
        config.numberOfSentences = 1
        flagLimit = False

    for sentenceInd in np.arange(config.numberOfSentences):
        if flagLimit:
            all_features = limitFeaturesSize(all_features2,config.numberOfSentences-sentenceInd)
        else:
            all_features = all_features2

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
        percentiles, mpr, recepricals, mrr, hit_rates, hit_rate = evaluate_wiki_recos(recos, output_path, gt_path, examples=examples)
        np.save('statPerLimit'+str(sentenceInd),(percentiles, mpr, recepricals, mrr, hit_rates, hit_rate))
        print(mrr)
        print(hit_rate)
        print(np.mean(list(filter(lambda p:p>0,percentiles.values()))))
        metrics = {"mrr": mrr, "mpr": mpr, "hit_rates": hit_rate}
    return recos, metrics

def vectorize_reco_average_search(all_features, titles,gt_path,config, output_path=""):
    gt = pickle.load(open(gt_path, "rb"))
    to_reco_indices = [index_amp(titles, title) for title in gt.keys()]
    to_reco_indices = list(filter(lambda title: title != None, to_reco_indices))

    averageEmb = []
    for articleFeature in all_features:
        averageEmb.append(np.average(articleFeature[0],0))
    averageEmb = np.array(averageEmb)
    averageEmb = torch.from_numpy(averageEmb)
    recos = []
    for i in tqdm(to_reco_indices):
        if i > len(all_features):
            print(f"GT title {titles[i]} was not evaluated")
            continue

        to_reco_flattened = averageEmb[i].reshape(-1,1)

        sim = sim_matrix(to_reco_flattened.T, averageEmb)
        recos.append((i, sim[0].argsort(descending=True)[1:]))

    examples = [[None, title] for title in titles]  # reco_utils compatibale
    percentiles, mpr, recepricals, mrr, hit_rates, hit_rate  = evaluate_wiki_recos(recos, output_path, gt_path, examples=examples)

    metrics = {"mrr": mrr, "mpr": mpr, "hit_rates": hit_rate}
    return recos, metrics

def hasGamePlaySection(data,sectionName='Gameplay.'):
    gameplayTitleSet = set()
    for sections,title in data:
        for section in sections:
            if section[1].split(':')[-1]==sectionName:
                gameplayTitleSet.add(title)
    return gameplayTitleSet

def limitFeaturesSize(all_features,index= 10):
    all_featuresOther = all_features
    for it in np.arange(len(all_featuresOther)):
        all_featuresOther[it][0] = all_features[it][0][:index]
    return all_featuresOther

def extractGamePlayFeatures(sectionString,all_features):
    featureVec = []
    for ind in np.arange(len(all_features)):
        L1 = len(all_features[ind])
        L2 = len(sectionString[ind][0])
        index = 0
        indCnt = 0
        for section in sectionString[ind][0]:
            if section[1].split(':')[-1]=='Gameplay.':
                index = indCnt
            indCnt+=1
        if L1==L2 or indCnt<L1:
            featureVec.append([all_features[ind][index]])
        else:
            featureVec.append([all_features[ind][-1]])
    return featureVec