import torch
from torch.utils.data import Dataset, DataLoader
from typing import NoReturn, Dict, List, Any, Tuple

from .triplet_generator import TripletGenerator

import logging


class PaperDataset(Dataset):
    def __init__(self, path: str):
        super(PaperDataset, self).__init__()
        data_dict = torch.load(path)
        self.paper_ids = data_dict["paper_ids"]
        self.encoded = data_dict["encoded"]
        self.network = data_dict["network"]

        assert all(
            tensor.size(0) == self.encoded["input_ids"].size(0)
            for tensor in self.encoded.values()
        )

        self.triplet_generator = TripletGenerator(self.paper_ids, self.network, 5)
        self.triplets = []
        for triplet in self.triplet_generator.generate_triplets():
            # print(triplet)
            self.triplets.append(triplet)

    def __getitem__(
        self, index: int
    ) -> Tuple[
        Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor]
    ]:
        triplet = self.triplets[index]
        # print(triplet)
        query = {k: v[triplet[0]] for k, v in self.encoded.items()}
        pos = {k: v[triplet[1]] for k, v in self.encoded.items()}
        neg = {k: v[triplet[2]] for k, v in self.encoded.items()}

        return query, pos, neg

    def __len__(self) -> int:
        return len(self.triplets)


if __name__ == "__main__":
    dataset = PaperDataset("./train_file.pth")
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    for q, p, n in dataloader:
        print(q)
        print(p)
        print(n)

        break