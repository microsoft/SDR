import ast
from data.data_utils import get_gt_seeds_titles, raw_data_link
import nltk
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
import os
import pickle
import numpy as np
from tqdm import tqdm
import torch
import json
import csv
import sys
from models.reco.recos_utils import index_amp


nltk.download("punkt")


class ClusterSummaryDatasetSentences(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, hparams, dataset_name, block_size, mode="train"):
        self.hparams = hparams
        cached_features_file = os.path.join(
            f"data/datasets/cached_proccessed/{dataset_name}",
            f"bs_{block_size}_{dataset_name}_{type(self).__name__}_tokenizer_{str(type(tokenizer)).split('.')[-1][:-2]}_mode_{mode}",
        )
        self.cached_features_file = cached_features_file
        os.makedirs(os.path.dirname(cached_features_file), exist_ok=True)

        raw_data_path = f"data/datasets/{dataset_name}/raw_data"


        all_articles = self.save_load_splitted_dataset(mode, cached_features_file, raw_data_path)
        if mode=='test':
            if hparams.gt_task=="catalog":
                queryGt = []
            else:
                queryGt = list(pickle.load(open('/home/jonathanE/Desktop/Github/SDR/SDR/data/datasets/video_games/video_games_gt.dict','rb')).keys())
        self.hparams = hparams

        max_article_len,max_sentences, max_sent_len = int(1e6), 16, 10000
        block_size = min(block_size, tokenizer.max_len_sentences_pair) if tokenizer is not None else block_size
        self.block_size = block_size
        self.tokenizer = tokenizer

        if  os.path.exists(cached_features_file) and (self.hparams is None or not self.hparams.overwrite_data_cache):#
            print("\nLoading features from cached file %s", cached_features_file)
            with open(cached_features_file, "rb") as handle:
                self.examples, self.indices_map = pickle.load(handle)
        else:
            print("\nCreating features from dataset file at ", cached_features_file)

            self.examples = []
            self.indices_map = []
            if mode=='test':
                enumerateData  = queryGt+all_articles
            else:
                enumerateData = all_articles
            for idx_article, article in enumerate(tqdm(enumerateData)):
                this_sample_sections = []
                if dataset_name=='msdocs':
                    title, sections = article[0], article[1]
                elif (mode=='test') and  ((dataset_name=='video_games_cluster')or self.hparams.dataset_name =='video_games_cluster_multianchor') and idx_article<len(queryGt):
                    title, sections = article, [article]
                #ArgName
                elif True:
                    sectionLidts = set(['Gameplay','Plot','Reception','Release'])
                    title, sections = article[0], [(article[it] , article[it+1]) for it in range(1,len(article),2) if article[it] in sectionLidts]
                else:
                    title, sections = article[0].replace(':',''),[article[0] +' ' + article[1]]
                    title, sections = article[0], [article[0] +' ' + article[1]]
                valid_sections_count = 0
                savedSection = ""
                for section_idx, section in enumerate(sections):
                    if section[1] == "":
                        continue
                    #if section=='':
                    #    section = title
                    this_sections_sentences = []                    
                    valid_sentences_count = 0
                    title_with_base_title = "{}:{}".format(title, section[0])

                    tokenized_sentences = nltk.sent_tokenize(section[1][:max_article_len])
                    #if len(tokenized_sentences)>max_sentences:
                    #    print(title)
                    #if len(tokenized_sentences)==0:
                    #    print(title)
                    for sent_idx, sent in enumerate(tokenized_sentences[:max_sentences]):
                        tokenized_desc = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(json.dumps(sent[:max_sent_len])))[
                            :block_size
                        ]
                        this_sections_sentences.append(
                            (
                                tokenized_desc,
                                len(tokenized_desc),
                                idx_article,
                                valid_sections_count,
                                valid_sentences_count,
                                sent[:max_sent_len],
                            ),
                        )
                        self.indices_map.append((idx_article, valid_sections_count, valid_sentences_count))
                        valid_sentences_count += 1
                    this_sample_sections.append((this_sections_sentences, title_with_base_title))
                    valid_sections_count += 1
                self.examples.append((this_sample_sections, title))
            print("\nSaving features into cached file %s", cached_features_file)
            with open(cached_features_file, "wb") as handle:
                pickle.dump((self.examples, self.indices_map), handle, protocol=pickle.HIGHEST_PROTOCOL)
        self.labels = [idx_article for idx_article, _, _ in self.indices_map]
    
    def save_load_splitted_dataset(self, mode, cached_features_file, raw_data_path):
        proccessed_path = f"{cached_features_file}_EXAMPLES"
        #debug - only when new dataset is added and need to parse it
        
        if self.hparams.summaryFlag and self.hparams.dataset_name =='video_games_cluster_multianchor':
            sectionLidts = set(['Gameplay\n','Plot\n','Reception\n','Release\n'])
            cached_features_file+='multiAnchor'
            proccessed_path = f"{cached_features_file}_EXAMPLES"
            file2 = open(raw_data_path, "w",encoding='utf8')

            openFile = '/raid/itzik/text_rank_output/latest_multi_anchor_stable_sent_to_jonathan/cohrent_order/'
            lISToFfILES = os.listdir(openFile)
            for it in lISToFfILES:
                fileRoot = openFile+"/"+it
                f = open(fileRoot, "r",encoding='utf8')
                fileContent = f.readlines()
                sectionInd = 0
                file2.write(it)
                while sectionInd<len(fileContent):
                    if fileContent[sectionInd] in sectionLidts:
                        file2.write( "%s" %'\t' + fileContent[sectionInd][:-1] +'\t'+fileContent[sectionInd+1][:-1])
                        sectionInd+=2
                    else:
                        sectionInd+=1
                file2.writelines("\n")
        if not os.path.exists(proccessed_path):
            all_articles = self.read_all_articles(raw_data_path)
            indices = list(range(len(all_articles)))
            if mode != "test":
                train_indices = sorted(
                    np.random.choice(indices, replace=False, size=int(len(all_articles) * self.hparams.train_val_ratio))
                )
                val_indices = np.setdiff1d(list(range(len(all_articles))), train_indices)
                indices = train_indices if mode == "train" else val_indices

            articles = []
            for i in indices:
                articles.append(all_articles[i])
            all_articles = articles
            pickle.dump(all_articles, open(proccessed_path, "wb"))
            print(f"\nsaved dataset at {proccessed_path}")
        else:
            all_articles = pickle.load(open(proccessed_path, "rb"))
        setattr(self.hparams, f"{mode}_data_file", proccessed_path)
        return all_articles

    def read_all_articles(self, raw_data_path):
        csv.field_size_limit(sys.maxsize)
        with open(raw_data_path, newline="") as f:
            reader = csv.reader(f,delimiter ='\t')
            all_articles = list(reader)
        return all_articles

    def __len__(self):
        return len(self.indices_map)

    def __getitem__(self, item):
        idx_article, idx_section, idx_sentence = self.indices_map[item]
        sent = self.examples[idx_article][0][idx_section][0][idx_sentence]

        return (
            torch.tensor(self.tokenizer.build_inputs_with_special_tokens(sent[0]), dtype=torch.long,)[
                : self.hparams.limit_tokens
            ],
            self.examples[idx_article][1],
            self.examples[idx_article][0][idx_section][1],
            sent[1],
            idx_article,
            idx_section,
            idx_sentence,
            item,
            self.labels[item],
        )

class ClusterSummaryDatasetSentencesTest(ClusterSummaryDatasetSentences):
    def __init__(self, tokenizer: PreTrainedTokenizer, hparams, dataset_name, block_size, mode="test"):
        super().__init__(tokenizer, hparams, dataset_name, block_size, mode=mode)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        sections = []
        for idx_section, section in enumerate(self.examples[item][0]):
            sentences = []
            for idx_sentence, sentence in enumerate(section[0]):
                sentences.append(
                    (
                        torch.tensor(self.tokenizer.build_inputs_with_special_tokens(sentence[0]), dtype=torch.long,),
                        self.examples[item][1],
                        section[1],
                        sentence[1],
                        item,
                        idx_section,
                        idx_sentence,
                        item,
                        self.labels[item],
                    )
                )
            sections.append(sentences)
        return sections
