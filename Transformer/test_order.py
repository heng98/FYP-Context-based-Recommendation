import torch
from torch.utils.data import Dataset, DataLoader
from utils import distributed
from torch.utils.data.distributed import DistributedSampler

import argparse


class SampleDataset(Dataset):
    def __init__(self) -> None:
        super(SampleDataset, self).__init__()

        self.data = list(range(128))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        return self.data[index]


parser = argparse.ArgumentParser()
parser.add_argument('--test')

config = parser.parse_args()


distributed.init_distributed_mode(config)
sample = SampleDataset()
sampler = DistributedSampler(sample, shuffle=False)
dataloader = DataLoader(sample, batch_size=4, sampler=sampler)

for i in dataloader:
    print(i)