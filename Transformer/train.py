import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from model.embedding_model import EmbeddingModel
from model.triplet_loss import TripletLoss
from data.dataset import PaperDataset

from tqdm import tqdm
import argparse
import json
import logging

logger = logging.getLogger()

if __name__ == "__main__":

    config = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = EmbeddingModel(config).to(device)
    criterion = TripletLoss().to(device)

    dataset = PaperDataset("./train_file.pth")
    logger.info(f"{len(dataset)} triplets is generated")
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=config.lr)

    for epoch in range(config.epochs):
        model.train()
        for q, p, n in tqdm(dataloader):
            optimizer.zero_grad()
            query_embedding = model(q)
            positive_embedding = model(p)
            negative_embedding = model(n)

            loss = criterion(query_embedding, positive_embedding, negative_embedding)
            loss.backward()

            optimizer.step()

    # parser = argparse.ArgumentParser()
    # parser.add_argument('--dataset_path', type=str, required=True)
    # parser.add_argument('--samples_per_query', type=int, default=5)

    # args = parser.parse_args()

    # train_set = process_dataset(args.dataset_path)
    # triplet_generator = TripletGenerator(train_set.keys(), train_set, args.samples_per_query)

    # dataset = PaperDataset()
    # model = embedding_model()

    # dataloader = DataLoader(dataset, batch_size=, shuffle=True, drop_last=True)
