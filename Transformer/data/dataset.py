import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info
from typing import Dict, List, Any, Set, Optional

from itertools import cycle

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


class TripletIterableDataset(IterableDataset):
    def __init__(
        self,
        dataset: Dict[str, Dict[str, Any]],
        query_paper_ids: List[str],
        candidate_papers_ids: Set[str],
        triplets_per_epcoh: int,
        config
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
            self.dataset,
            self.query_paper_ids,
            self.candidate_papers_ids,
            self.config
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

def worker_fn(worker_id):
    worker_info = get_worker_info()
    num_workers = worker_info.num_workers
    dataset = worker_info.dataset
    size = len(dataset.query_paper_ids) // num_workers + 1

    dataset.query_paper_ids = dataset.query_paper_ids[
        worker_id * size : (worker_id + 1) * size
    ]

if __name__ == "__main__":
    from torch.utils.data import DataLoader, get_worker_info
    import json
    import torch
    import random
    import transformers
    from triplet_generator import TripletGenerator

    with open("./DBLP_train_test_dataset_1.json", "r") as f:
        data_json = json.load(f)
        dataset_name = data_json["name"]
        train_dataset = data_json["train"]
        val_dataset = data_json["valid"]

    tokenizer = transformers.AutoTokenizer.from_pretrained("allenai/scibert_scivocab_cased")
    collater = TripletCollator(tokenizer, 256)
    config = Config()    
    query_paper_ids = list(train_dataset.keys())

    dataset = TripletIterableDataset(
        train_dataset,
        query_paper_ids[:1],
        set(query_paper_ids),
        config
    )
    triplet_generator = TripletGenerator(
        train_dataset,
        query_paper_ids[:1],
        set(query_paper_ids),
        config
    ).generate_triplets()

    dataloader = DataLoader(dataset, batch_size=8, num_workers=1, collate_fn=collater)
    # dataset.triplet_generator.update_nn_hard(torch.randn(len(train_dataset), 5), list(query_paper_ids))
    
    print(len(dataloader))
    # for j, i in enumerate(dataloader):
    #     print(j)
        

