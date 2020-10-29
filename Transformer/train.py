import torch
from torch.utils.data import DataLoader

from model.embedding_model import EmbeddingModel
from model.triplet_loss import TripletLoss


from data.dataset import PaperDataset

import argparse
import json
import logging



if __name__ == '__main__':
    dataset = PaperDataset('./train_file.pth')
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    for q, p, n in dataloader:
        print(q)
        print(p)
        print(n)

        break
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--dataset_path', type=str, required=True)
    # parser.add_argument('--samples_per_query', type=int, default=5)

    # args = parser.parse_args()

    # train_set = process_dataset(args.dataset_path)
    # triplet_generator = TripletGenerator(train_set.keys(), train_set, args.samples_per_query)








    # dataset = PaperDataset()
    # model = embedding_model()
    

    # dataloader = DataLoader(dataset, batch_size=, shuffle=True, drop_last=True)

