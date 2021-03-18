import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info
from typing import Dict, List, Any, Set, Optional

from .triplet_generator import TripletGenerator


class PaperPosDataset(Dataset):
    def __init__(
        self, dataset: List[Dict[str, Any]], candidate_paper: Dict[str, int], tokenizer, abstract=True
    ):
        super(PaperPosDataset, self).__init__()
        self.dataset = dataset
        self.candidate_paper = candidate_paper
        self.candidate_paper_set = set(candidate_paper)
        self.tokenizer = tokenizer

        self.abstract = abstract

        self.cache = {}

    def __getitem__(self, index: int):
        data = self.dataset[index]
        if self.abstract:
            encoded = self.tokenizer(
                data["title"],
                data["abstract"],
                padding="max_length",
                max_length=256,
                truncation=True,
                return_tensors="pt",
            )
        else:
            encoded = self.tokenizer(
                data["title"],
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


class TripletCollator:
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


class TripletIterableDataset(IterableDataset):
    def __init__(
        self,
        dataset: Dict[str, Dict[str, Any]],
        query_paper_ids: List[str],
        candidate_papers_ids: Set[str],
        config
    ):
        super().__init__()
        self.dataset = dataset
        self.query_paper_ids = query_paper_ids
        self.candidate_papers_ids = candidate_papers_ids
        self.config = config

        self.triplet_generator = self._build_triplet_generator()
 
    def _build_triplet_generator(self):
        triplet_generator = TripletGenerator(
            self.dataset,
            self.query_paper_ids,
            self.candidate_papers_ids,
            self.config
        )
        return triplet_generator

    def __iter__(self):
        worker_info = get_worker_info()
        mod = worker_info.num_workers
        shift = worker_info.id
        # triplet_generator = self._build_triplet_generator()
        for i, triplet in enumerate(self.triplet_generator.generate_triplets()):
            if (i + shift) % mod == 0:
                query_paper = self.dataset[triplet[0]]
                pos_paper = self.dataset[triplet[1]]
                neg_paper = self.dataset[triplet[2]]

                yield query_paper, pos_paper, neg_paper 


class DistributedTripletIterableDataset(TripletIterableDataset):
    def __init__(
        self,
        config,
        query_paper_ids_idx_mapping: Dict[str, int],
        candidate_papers_ids: Set[str],
        dataset: List[Dict[str, Any]],
        samples_per_query: int,
        ratio_hard_neg: Optional[float] = 0.5,
    ):
        super().__init__(
            query_paper_ids_idx_mapping,
            candidate_papers_ids,
            dataset,
            samples_per_query,
            ratio_hard_neg,
        )
        self.rank = config.rank
        self.world_size = config.world_size

        self.mod = -1
        self.shift = -1

    def __iter__(self):
        pass


class Config:
    samples_per_query = 5
    ratio_hard_neg = 0.5
    ratio_nn_neg = 0.1


if __name__ == "__main__":
    from torch.utils.data import DataLoader, get_worker_info
    import json
    import torch
    from transformers import AutoTokenizer


    with open("./DBLP_train_test_dataset_1.json", "r") as f:
        data_json = json.load(f)
        dataset_name = data_json["name"]
        train_dataset = data_json["train"]
        val_dataset = data_json["valid"]


    config = Config()    
    query_paper_ids = list(train_dataset.keys())

    dataset = TripletIterableDataset(
        train_dataset,
        query_paper_ids,
        set(query_paper_ids),
        config
    )

    print(query_paper_ids[:4])
    # dataloader = DataLoader(dataset, num_workers=4, worker_init_fn=worker_fn)
    # dataset.triplet_generator.update_nn_hard(torch.randn(len(train_dataset), 5), list(query_paper_ids))
    for i in dataloader:
        continue
        # print(i[0]["ids"])
        # a = [j["ids"][0] for j in i]
        # print(a)
