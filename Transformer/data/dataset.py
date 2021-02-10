import torch
from torch.utils.data import Dataset, IterableDataset
from typing import NoReturn, Dict, List, Any, Tuple, Set, Optional

from .triplet_generator import TripletGenerator


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


# class TripletIterableDataset(IterableDataset):
#     def __init__(
#         self,
#         dataset: List[Dict[str, Any]],
#         query_paper_ids_idx_mapping: Dict[str, int],
#         candidate_paper_ids_idx_mapping: Dict[str, int],
#         samples_per_query: int,
#         ratio_hard_neg: Optional[float] = 0.5,
#     ):
#         super(TripletIterableDataset, self).__init__()

#         self.dataset = dataset
#         self.query_paper_ids_idx_mapping = query_paper_ids_idx_mapping
#         self.candidate_paper_ids_idx_mapping = candidate_paper_ids_idx_mapping

#         self.triplet_generator = TripletGenerator(
#             list(self.query_paper_ids_idx_mapping.values()),
#             set(self.query_paper_ids_idx_mapping),
#             dataset,
#             samples_per_query,
#             ratio_hard_neg,
#         )

#     def __iter__(self):
#         for triplet in self.triplet_generator.generate_triplets():
#             query_idx = self.query_paper_ids_idx_mapping[triplet[0]]
#             pos_idx = self.candidate_paper_ids_idx_mapping[triplet[1]]
#             neg_idx = self.candidate_paper_ids_idx_mapping[triplet[2]]

#             yield self.dataset[query_idx], self.dataset[pos_idx], self.dataset[neg_idx]


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


# class PaperDataset:
#     def __init__(self, data_path: str, encoded_path: str, config):
#         data_dict = torch.load(data_path)

#         self.paper_ids_idx_mapping = data_dict["paper_ids_idx_mapping"]
#         self.network = data_dict["network"]

#         self.encoded = torch.load(encoded_path)["encoded"]
#         self.config = config

#         assert all(
#             tensor.size(0) == self.encoded["input_ids"].size(0)
#             for tensor in self.encoded.values()
#         )

#     def get_paper_pos_dataset(self, candidate_paper_ids_idx_mapping):
#         return PaperPosDataset(
#             self.paper_ids_idx_mapping,
#             self.encoded,
#             self.network,
#             candidate_paper_ids_idx_mapping,
#         )

#     def get_triplet_dataset(self, candidate_paper_ids_idx_mapping):
#         return TripletIterableDataset(
#             self.paper_ids_idx_mapping,
#             self.encoded,
#             self.network,
#             candidate_paper_ids_idx_mapping,
#             self.config.samples_per_query,
#             self.config.ratio_hard_neg
#         )
