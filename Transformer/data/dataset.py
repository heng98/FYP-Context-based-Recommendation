import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info
from typing import Dict, List, Any, Set, Optional

import re

from .triplet_generator import TripletGenerator


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


class QueryPairDataset(Dataset):
    def __init__(self, triplet_list, embedding):
        super(QueryPairDataset, self).__init__()

        query_pairs = []
        label = []

        for q, p, n in triplet_list:
            query_pairs.extend([(q, p), (q, n)])
            label.extend([1, 0])

        assert len(query_pairs) == len(label)

        self.query_pairs = query_pairs
        self.label = label
        self.embedding = embedding
        # self.dataset = dataset

    def __getitem__(self, index: int):
        query_idx, candidate_idx = self.query_pairs[index]
        query = self.embedding[query_idx]
        candidate = self.embedding[candidate_idx]
        label = self.label[index]

        return query, candidate, label

    def __len__(self) -> int:
        return len(self.query_pairs)


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
        query_title = [data[0]["title"] for data in batch]
        query_abstract = [data[0]["abstract"] for data in batch]

        pos_title = [data[1]["title"] for data in batch]
        pos_abstract = [data[1]["abstract"] for data in batch]

        neg_title = [data[2]["title"] for data in batch]
        neg_abstract = [data[2]["abstract"] for data in batch]

        query_encoded = self._encode(query_title, query_abstract)
        pos_encoded = self._encode(pos_title, pos_abstract)
        neg_encoded = self._encode(neg_title, neg_abstract)

        return query_encoded.data, pos_encoded.data, neg_encoded.data

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
                result.append(
                    len(set(t1).intersection(set(t2))) / union
                )
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
    ):
        super().__init__()
        self.dataset = dataset
        self.query_paper_ids = query_paper_ids
        self.candidate_papers_ids = candidate_papers_ids
        self.triplets_per_epoch = triplets_per_epcoh
        self.config = config

        self.triplet_generator = self._build_triplet_generator()

        self._yielded = 0

    def _build_triplet_generator(self):
        triplet_generator = TripletGenerator(
            self.dataset, self.query_paper_ids, self.candidate_papers_ids, self.config
        )
        return triplet_generator.generate_triplets()

    def __iter__(self):
        return self

    def __next__(self):
        if self._yielded == len(self):
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

        return query_paper, pos_paper, neg_paper

    def __len__(self):
        return self.triplets_per_epoch
