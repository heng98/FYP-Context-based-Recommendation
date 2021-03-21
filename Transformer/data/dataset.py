import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info

import random
from typing import Dict, List, Any, Set
import re

from .triplet_generator import TripletGenerator
from .preprocessor import DefaultPreprocessor


class PaperPosDataset(Dataset):
    def __init__(
        self, dataset: List[Dict[str, Any]], candidate_paper: Dict[str, int], tokenizer
    ):
        super(PaperPosDataset, self).__init__()
        self.dataset = dataset
        self.candidate_paper = candidate_paper
        self.candidate_paper_set = set(candidate_paper)
        self.tokenizer = tokenizer

        self.mapping = list(dataset.keys())

        self.cache = {}

    def __getitem__(self, index: int):
        data = self.dataset[self.mapping[index]]
        encoded = self.tokenizer(
            data["title"],
            data["abstract"],
            padding="max_length",
            max_length=256,
            truncation=True,
            return_tensors="pt",
        )

        if index not in self.cache:
            positive_candidates = set(data["pos"]) & self.candidate_paper_set
            self.cache[index] = positive_candidates
        else:
            positive_candidates = self.cache[index]

        return encoded, [self.candidate_paper[ids] for ids in positive_candidates]

    def __len__(self) -> int:
        return len(self.dataset)


class TripletDataset(Dataset):
    def __init__(self, triplet_list, dataset):
        super(TripletDataset, self).__init__()

        self.triplet_list = triplet_list
        self.dataset = dataset

    def __getitem__(self, index: int):
        triplet = self.triplet_list[index]
        query = self.dataset[triplet[0]]
        pos = self.dataset[triplet[1]]
        neg = self.dataset[triplet[2]]

        return query, pos, neg

    def __len__(self) -> int:
        return len(self.triplet_list)


class TripletCollater:
    def __init__(self, tokenizer, max_seq_len):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __call__(self, batch):
        query_title = [data["query_paper"]["title"] for data in batch]
        query_abstract = [data["query_paper"]["abstract"] for data in batch]

        pos_title = [data["pos_paper"]["title"] for data in batch]
        pos_abstract = [data["pos_paper"]["abstract"] for data in batch]

        neg_title = [data["neg_paper"]["title"] for data in batch]
        neg_abstract = [data["neg_paper"]["abstract"] for data in batch]

        margin = [data["margin"] for data in batch]
        margin = torch.tensor(margin, dtype=torch.float32)

        encoded_query = self._encode(query_title, query_abstract)
        encoded_pos = self._encode(pos_title, pos_abstract)
        encoded_neg = self._encode(neg_title, neg_abstract)

        return {
            "encoded_query": encoded_query,
            "encoded_positive": encoded_pos,
            "encoded_negative": encoded_neg,
            "margin": margin,
        }

    def _encode(self, title: List[str], abstract: List[str]):
        return self.tokenizer(
            title,
            abstract,
            padding="max_length",
            max_length=self.max_seq_len,
            truncation=True,
            return_tensors="pt",
        )


class TripletRankerCollater:
    def __init__(self):
        # self.tokenizer = tokenizer
        # self.max_seq_len = max_seq_len
        self.pattern = re.compile(r"[\w']+")

    def __call__(self, batch):
        # query = [data[0]["title"] + data[0]["abstract"] for data in batch]
        # pos = [data[1]["title"] + data[1]["abstract"] for data in batch]
        # neg = [data[2]["title"] + data[2]["abstract"] for data in batch]

        # query_encoded = self._encode(query).data
        # pos_encoded = self._encode(pos).data
        # neg_encoded = self._encode(neg).data

        # pos_encoded["input_ids"][:, 0] = self.tokenizer.sep_token_id
        # neg_encoded["input_ids"][:, 0] = self.tokenizer.sep_token_id

        # keys = query_encoded.keys()
        # query_pos_encoded = dict()
        # query_neg_encoded = dict()

        # for k in keys:
        #     query_pos_encoded[k] = torch.cat([query_encoded[k], pos_encoded[k]], axis=1)
        #     query_neg_encoded[k] = torch.cat([query_encoded[k], neg_encoded[k]], axis=1)

        # return query_pos_encoded, query_neg_encoded

        query_ids = [data[0]["ids"] for data in batch]
        pos_ids = [data[1]["ids"] for data in batch]
        neg_ids = [data[2]["ids"] for data in batch]

        query = [data[0]["abstract"] for data in batch]
        pos = [data[1]["abstract"] for data in batch]
        neg = [data[2]["abstract"] for data in batch]

        query_pos_jaccard = torch.tensor([self._jaccard(query, pos)]).T
        query_neg_jaccard = torch.tensor([self._jaccard(query, neg)]).T

        return query_ids, pos_ids, neg_ids, query_pos_jaccard, query_neg_jaccard

    # def _encode(self, text: List[str]):
    #     return self.tokenizer(
    #         text,
    #         padding="max_length",
    #         max_length=self.max_seq_len // 2,
    #         truncation=True,
    #         return_tensors="pt",
    #     )

    def _jaccard(self, text_1, text_2):
        result = []
        tokenized_t1 = [self.pattern.findall(t1) for t1 in text_1]
        tokenized_t2 = [self.pattern.findall(t2) for t2 in text_2]

        for t1, t2 in zip(tokenized_t1, tokenized_t2):
            union = len(set(t1).union(set(t2)))
            if union > 0:
                result.append(len(set(t1).intersection(set(t2))) / union)
            else:
                result.append(0)

        return result


class TripletIterableDataset(IterableDataset):
    def __init__(
        self,
        dataset: Dict[str, Dict[str, Any]],
        query_paper_ids: List[str],
        candidate_papers_ids: Set[str],
        triplets_per_epcoh: int,
        config,
        nn_hard=False,
        doc_embedding=None,
        preprocessor=None,
    ):
        super().__init__()
        self.dataset = dataset
        self.query_paper_ids = query_paper_ids
        self.candidate_papers_ids = candidate_papers_ids
        self.triplets_per_epoch = triplets_per_epcoh
        self.config = config

        self.nn_hard = nn_hard
        self.doc_embedding = doc_embedding

        self.preprocessor = (
            preprocessor if preprocessor is not None else DefaultPreprocessor()
        )
        self._yielded = 0
        self.length = self.triplets_per_epoch

    def _build_triplet_generator(self):
        triplet_generator = TripletGenerator(
            self.dataset, self.query_paper_ids, self.candidate_papers_ids, self.config
        )
        if self.nn_hard:
            triplet_generator.update_nn_hard(self.doc_embedding)
        return triplet_generator.generate_triplets()

    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info is not None:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

            split_size = (len(self.query_paper_ids) // num_workers) + 1
            self.length //= num_workers
            self.query_paper_ids = self.query_paper_ids[worker_id * split_size: (worker_id + 1) * split_size]
        
        self.triplet_generator = self._build_triplet_generator()
        return self

    def __next__(self):
        if self._yielded >= len(self):
            self._yielded = 0
            raise StopIteration

        try:
            self._yielded += 1
            triplet = next(self.triplet_generator)

        except StopIteration:
            self.triplet_generator = self._build_triplet_generator()
            triplet = next(self.triplet_generator)

        query_paper = self.dataset[triplet[0]]
        pos_paper = self.dataset[triplet[1]]
        neg_paper = self.dataset[triplet[2]]
        margin = triplet[3]

        return self.preprocessor(query_paper, pos_paper, neg_paper, margin)

    def __len__(self):
        return self.length


# Dataset {query_ids:"", neg_ids: ["", ""]}
class GroupedTrainedDataset(Dataset):
    def __init__(self, args, corpus, dataset):
        super(GroupedTrainedDataset, self).__init__()
        self.args = args
        self.corpus = corpus
        self.dataset = dataset

    def __getitem__(self, index):
        query_id = self.dataset[index]["query_ids"]
        positive_ids = self.dataset[index]["positive_ids"]
        negative_ids = self.dataset[index]["negative_ids"]

        if len(positive_ids) == 0:
            positive_ids = self.corpus[query_id]["citations"]

        query_abstract = self.corpus[query_id]["abstract"]
        positive_abstract = self.corpus[random.choice(positive_ids)]["abstract"]

        if len(negative_ids) < self.args.group_size - 1:
            negative_abstracts = [
                self.corpus[neg_id]["abstract"]
                for neg_id in random.choices(negative_ids, k=self.args.group_size - 1)
            ]
        else:
            negative_abstracts = [
                self.corpus[neg_id]["abstract"]
                for neg_id in random.sample(negative_ids, k=self.args.group_size - 1)
            ]

        group = [(query_abstract, positive_abstract)] + [
            (query_abstract, negative_abstract)
            for negative_abstract in negative_abstracts
        ]

        return group

    def __len__(self):
        return len(self.dataset)


class GroupedTrainedCollater:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        query_abstract = [data[0] for group in batch for data in group]
        candidate_abstract = [data[1] for group in batch for data in group]

        return self._encode(query_abstract, candidate_abstract)

    def _encode(self, query_abstract: List[str], candidate_abstract: List[str]):
        return self.tokenizer(
            query_abstract,
            candidate_abstract,
            padding="max_length",
            max_length=self.max_seq_len,
            truncation="only_second",
            return_tensors="pt",
        )
