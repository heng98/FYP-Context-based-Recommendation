import torch
from torch.utils.data import Dataset, DataLoader
from typing import NoReturn, Dict, List, Any, Tuple, Set, Optional

from .triplet_generator import TripletGenerator


class PaperPosDataset(Dataset):
    def __init__(
        self,
        paper_ids_idx_mapping: Dict[str, int],
        encoded: Dict[str, torch.Tensor],
        network: Dict[str, List[str]],
        candidate_paper_ids_idx_mapping: Dict[str, int],
    ):
        super(PaperPosDataset, self).__init__()

        self.paper_ids_idx_mapping = paper_ids_idx_mapping
        self.encoded = encoded
        self.network = network
        self.candidate_paper_ids_idx_mapping = candidate_paper_ids_idx_mapping

        self.paper_ids_list = list(self.paper_ids_idx_mapping.keys())

    def __getitem__(self, index: int):
        encoded = {k: v[index] for k, v in self.encoded.items()}
        paper_ids = self.paper_ids_list[index]
        positive_candidates = set(self.network[paper_ids]["pos"]) & set(
            self.candidate_paper_ids_idx_mapping.keys()
        )
        pos = [self.candidate_paper_ids_idx_mapping[ids] for ids in positive_candidates]

        return encoded, pos

    def __len__(self) -> int:
        return len(self.paper_ids_list)


class TripletDataset(Dataset):
    def __init__(
        self,
        query_paper_ids_idx_mapping: Dict[str, int],
        encoded: Dict[str, torch.Tensor],
        network: Dict[str, List[str]],
        candidate_paper_ids_idx_mapping: Dict[str, int],
        samples_per_query: int,
        ratio_hard_neg: Optional[float] = 0.5,
    ):
        super(TripletDataset, self).__init__()

        self.query_paper_ids_idx_mapping = query_paper_ids_idx_mapping
        self.encoded = encoded
        self.network = network
        self.candidate_paper_ids_idx_mapping = candidate_paper_ids_idx_mapping

        self.paper_ids_list = list(self.query_paper_ids_idx_mapping.keys())

        self.triplet_generator = TripletGenerator(
            self.paper_ids_list,
            set(candidate_paper_ids_idx_mapping.keys()),
            self.network,
            samples_per_query,
            ratio_hard_neg=ratio_hard_neg,
        )

        self.triplet = []
        for triplet in self.triplet_generator.generate_triplets():
            self.triplet.append(triplet)

        self.triplet = list(set(self.triplet))

    def __getitem__(self, index: int):
        triplet = self.triplet[index]

        query_idx = self.query_paper_ids_idx_mapping[triplet[0]]
        pos_idx = self.candidate_paper_ids_idx_mapping[triplet[1]]
        neg_idx = self.candidate_paper_ids_idx_mapping[triplet[2]]

        query = {k: v[query_idx] for k, v in self.encoded.items()}
        pos = {k: v[pos_idx] for k, v in self.encoded.items()}
        neg = {k: v[neg_idx] for k, v in self.encoded.items()}

        return query, pos, neg

    def __len__(self) -> int:
        return len(self.triplet)


class PaperDataset:
    def __init__(self, data_path: str, encoded_path: str, config):
        data_dict = torch.load(data_path)

        self.paper_ids_idx_mapping = data_dict["paper_ids_idx_mapping"]
        self.network = data_dict["network"]

        self.encoded = torch.load(encoded_path)["encoded"]
        self.config = config

        assert all(
            tensor.size(0) == self.encoded["input_ids"].size(0)
            for tensor in self.encoded.values()
        )

    def get_paper_pos_dataset(self, candidate_paper_ids_idx_mapping):
        return PaperPosDataset(
            self.paper_ids_idx_mapping,
            self.encoded,
            self.network,
            candidate_paper_ids_idx_mapping,
        )

    def get_triplet_dataset(self, candidate_paper_ids_idx_mapping):
        return TripletDataset(
            self.paper_ids_idx_mapping,
            self.encoded,
            self.network,
            candidate_paper_ids_idx_mapping,
            self.config.samples_per_query,
            self.config.ratio_hard_neg
        )


if __name__ == "__main__":
    train_dataset = PaperDataset("./train_file.pth", "./encoded.pth", None)
    test_dataset = PaperDataset("./test_file.pth", "./encoded.pth", None)

    train_candidate_paper_ids_idx_mapping = train_dataset.paper_ids_idx_mapping
    test_candidate_paper_ids_idx_mapping = {
        **train_dataset.paper_ids_idx_mapping,
        **test_dataset.paper_ids_idx_mapping,
    }

    # train_triplet_dataset = train_dataset.get_triplet_dataset(
    #     train_candidate_paper_ids_idx_mapping
    # )
    # test_triplet_dataset = test_dataset.get_triplet_dataset(
    #     test_candidate_paper_ids_idx_mapping
    # )

    test_paper_pos_dataset = test_dataset.get_paper_pos_dataset(
        train_candidate_paper_ids_idx_mapping
    )


    # print(test_triplet_dataset.triplet)
    # print(len(train_triplet_dataset), len(test_triplet_dataset))

    # for data in test_triplet_dataset:
    #     print(data)

    # for data in test_paper_pos_dataset:
    #     print(data[1])