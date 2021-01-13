import torch
from torch.utils.data import Dataset, DataLoader
from typing import NoReturn, Dict, List, Any, Tuple

from .triplet_generator import TripletGenerator

import logging


class PaperTripletDataset(Dataset):
    def __init__(self, path: str):
        super(PaperTripletDataset, self).__init__()
        data_dict = torch.load(path)
        self.paper_ids = data_dict["paper_ids"]
        self.encoded = data_dict["encoded"]
        self.network = data_dict["network"]

        self.idx_paper_ids = {i: ids for i, ids in enumerate(self.paper_ids)}

        assert all(
            tensor.size(0) == self.encoded["input_ids"].size(0)
            for tensor in self.encoded.values()
        )

        self.triplet_generator = TripletGenerator(
            self.paper_ids, self.idx_paper_ids, self.network, 5
        )
        self.triplets = []
        for triplet in self.triplet_generator.generate_triplets():
            self.triplets.append(triplet)

    def __getitem__(
        self, index: int
    ) -> Tuple[
        Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor]
    ]:
        triplet = self.triplets[index]

        query = {k: v[triplet[0]] for k, v in self.encoded.items()}
        pos = {k: v[triplet[1]] for k, v in self.encoded.items()}
        neg = {k: v[triplet[2]] for k, v in self.encoded.items()}

        return query, pos, neg

    def __len__(self) -> int:
        return len(self.triplets)


class PaperEvalDataset(Dataset):
    def __init__(self, path: str, train_idx_paper_ids):
        super(PaperEvalDataset, self).__init__()
        data_dict = torch.load(path)
        self.paper_ids = data_dict["paper_ids"]
        self.encoded = data_dict["encoded"]
        self.network = data_dict["network"]

        self.idx_paper_ids = {i: ids for i, ids in enumerate(self.paper_ids)}

        self.train_paper_ids_idx = {k: v for v, k in train_idx_paper_ids.items()}

        assert all(
            tensor.size(0) == self.encoded["input_ids"].size(0)
            for tensor in self.encoded.values()
        )

    def __getitem__(self, index: int) -> Tuple[Dict[str, torch.Tensor], List[int]]:
        query = {k: v[[index]] for k, v in self.encoded.items()}
        positives = [
            self.train_paper_ids_idx[i]
            for i in self.network[self.idx_paper_ids[index]]["pos"]
        ]

        return query, positives

    def __len__(self) -> int:
        return len(self.paper_ids)


# if __name__ == "__main__":
#     dataset = PaperDataset("./test_file.pth")
#     dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

#     for q, ids in dataset:
#         # print(q)
#         print(ids)
