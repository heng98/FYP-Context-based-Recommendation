import torch
from torch.utils.data import Dataset
from typing import Dict, List, Any


class PaperPosDataset(Dataset):
    def __init__(
        self,
        dataset: List[Dict[str, Any]],
        candidate_paper: Dict[str, int],
        tokenizer
    ):
        super(PaperPosDataset, self).__init__()
        self.dataset = dataset
        self.candidate_paper = candidate_paper
        self.candidate_paper_set = set(candidate_paper)
        self.tokenizer = tokenizer

        self.cache = {}
        
    def __getitem__(self, index: int):
        data = self.dataset[index]
        encoded = self.tokenizer(
            data["title"],
            data["abstract"],
            padding="max_length",
            max_length=512,
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


class QueryPairCollater:
    def __init__(self, embedding):
        self.embedding = embedding

    def __call__(self, batch):
        query_idx = [data[0] for data in batch]
        candidate_idx = [data[1] for data in batch]
        labels = [data[2] for data in batch]

        query_embedding = self.embedding[query_idx]
        candidate_embedding = self.embedding[candidate_idx]
        labels = torch.tensor(labels, dtype=torch.float)

        return query_embedding, candidate_embedding, labels

class TripletDataset(Dataset):
    def __init__(
        self,
        triplet_list,
        dataset
    ):
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


class TripletCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

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
            max_length=512,
            truncation=True,
            return_tensors="pt",
        )


