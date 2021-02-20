import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

import transformers

from model.embedding_model import EmbeddingModel
from model.reranker_model import SimpleReranker
from data.dataset import TripletDataset, TripletCollator
from utils import distributed

import json
import os
from tqdm import tqdm
import argparse
import random
import logging
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def train_one_epoch(
    embedding_model, model, train_dataloader, criterion, optimizer, scheduler, epoch, writer, device
):
    logger.info(f"===Training epoch {epoch}===")
    model.train()
    embedding_model.eval()
    for i, (q, p, n) in enumerate(
        tqdm(train_dataloader), epoch * len(train_dataloader)  # Need check
    ):
        q, p, n = (
            to_device_dict(q, device),
            to_device_dict(p, device),
            to_device_dict(n, device),
        )

        with torch.no_grad():
            query_embedding = embedding_model(q)
            positive_embedding = embedding_model(p)
            negative_embedding = embedding_model(n)


        pos_result = model(query_embedding, positive_embedding)
        neg_result = model(query_embedding, negative_embedding)
        loss = criterion(pos_result, neg_result, torch.ones(pos_result.size(0)))
        loss.backward()

        if (i + 1) % 50 == 0:
            loss_recorded = loss.detach().clone()
            writer.add_scalar("train_ranker/loss", loss_recorded.item(), i)

        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()


@torch.no_grad()
def eval(embedding_model, model, eval_dataloader, criterion, epoch, writer, device):
    embedding_model.eval()
    model.eval()
    loss_list = []
    num_correct = 0
    for q, p, n in tqdm(eval_dataloader):
        q, p, n = (
            to_device_dict(q, device),
            to_device_dict(p, device),
            to_device_dict(n, device),
        )

        query_embedding = embedding_model(q)
        positive_embedding = embedding_model(p)
        negative_embedding = embedding_model(n)


        pos_result = model(query_embedding, positive_embedding)
        neg_result = model(query_embedding, negative_embedding)
        loss = criterion(pos_result, neg_result, torch.ones(pos_result.size(0)))

        loss_list.append(loss_recorded.item())
   
    epoch_loss = torch.tensor(loss_list, dtype=torch.float).mean()
    # acc = num_correct / len(eval_dataloader.dataset)

    writer.add_scalar("val_ranker/loss", epoch_loss, epoch)
    # writer.add_scalar("val_ranker/acc", acc, epoch)

def to_device_dict(d, device):
    return {k: v.to(device) for k, v in d.items()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument(
        "--model_name", type=str, default="allenai/scibert_scivocab_cased"
    )
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch_size", type=int, default=8)

    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--experiment_name", type=str, required=True)
    parser.add_argument("--reload_epoch", type=int, required=True)

    config = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SimpleReranker().to(device)
    criterion = nn.MarginRankingLoss()
    # random.seed(config.seed)

    if not os.path.exists(f"weights/{config.experiment_name}"):
        os.makedirs(f"weights/{config.experiment_name}")

    with open("dblp_triplet.pkl", "rb") as f:
        unpickled_data = pickle.load(f)

    tokenizer = transformers.AutoTokenizer.from_pretrained(config.model_name)
    embedding_model = EmbeddingModel(config)
    state_dict = torch.load(
        f"weights/{config.experiment_name}/weights_{config.reload_epoch}.pth", 
        map_location="cuda:0"
    )["state_dict"]

    embedding_model.load_state_dict(state_dict)
    embedding_model = embedding_model.to(device)

    # embedding = torch.empty((len(unpickled_data["dataset"]), 768), dtype=torch.float)
    # with torch.no_grad():
    #     for i, data in enumerate(tqdm(unpickled_data["dataset"])):
    #         encoded = tokenizer(
    #             data["title"],
    #             data["abstract"],
    #             padding="max_length",
    #             max_length=512,
    #             truncation=True,
    #             return_tensors="pt",
    #         )
    #         encoded = to_device_dict(encoded, device)
    #         tmp_embedding = embedding_model(encoded)
    #         embedding[i] = tmp_embedding.cpu()

    # del embedding_model
    # torch.cuda.empty_cache()

    # torch.save(embedding, "all_embedding_dblp.pth")
    # embedding = torch.load("all_embedding_dblp.pth")

    train_triplet_dataset = TripletDataset(
        unpickled_data["train"], unpickled_data["dataset"]
    )
    test_triplet_dataset = TripletDataset(
        unpickled_data["valid"], unpickled_data["dataset"]
    )

    collater = TripletCollator(tokenizer, config.max_seq_len)

    train_triplet_dataloader = DataLoader(
        train_triplet_dataset,
        batch_size=config.batch_size,
        num_workers=2,
        pin_memory=True,
        shuffle=True,
        collate_fn=collater,
    )
    test_triplet_dataloader = DataLoader(
        test_triplet_dataset,
        batch_size=config.batch_size,
        num_workers=2,
        pin_memory=True,
        collate_fn=collater,
    )


    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer,
        len(train_triplet_dataloader),
        len(train_triplet_dataloader) * config.epochs,
    )

    writer = SummaryWriter(f"runs/{config.experiment_name}")

    for epoch in range(0, config.epochs):
        train_one_epoch(
            embedding_model,
            model,
            train_triplet_dataloader,
            criterion,
            optimizer,
            scheduler,
            epoch,
            writer,
            device
        )
        eval(embedding_model, model, test_triplet_dataloader, criterion, epoch, writer, device)

        torch.save(
            {"state_dict": model.state_dict()},
            f"weights/{config.experiment_name}/reranker_weights_{epoch}.pth",
        )
