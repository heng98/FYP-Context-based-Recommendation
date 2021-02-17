import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

import transformers

from model.embedding_model import EmbeddingModel
from model.reranker_model import SimpleReranker
from data.dataset import QueryPairDataset
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
    model, train_dataloader, criterion, optimizer, scheduler, epoch, writer, device
):
    logger.info(f"===Training epoch {epoch}===")
    model.train()
    for i, (q, c, label) in enumerate(
        tqdm(train_dataloader), epoch * len(train_dataloader)  # Need check
    ):
        q = q.to(device)
        c = c.to(device)
        label = label.float().unsqueeze(1).to(device)

        output = model(q, c)
        loss = criterion(output, label)
        loss.backward()

        if (i + 1) % 50 == 0:
            loss_recorded = loss.detach().clone()
            writer.add_scalar("train_ranker/loss", loss_recorded.item(), i)

        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()


@torch.no_grad()
def eval(model, eval_dataloader, criterion, epoch, writer, device):
    model.eval()
    loss_list = []
    num_correct = 0
    for i, (q, c, label) in enumerate(tqdm(eval_dataloader), epoch):
        q = q.to(device)
        c = c.to(device)
        label = label.float().unsqueeze(1).to(device)

        output = model(q, c)
        loss = criterion(output, label)

        output = output.sigmoid()
        prediction = (output > 0.5).float()
        num_correct += torch.sum(prediction == label).item()

        if i % 50 == 0:
            loss_list.append(loss.item())

    epoch_loss = torch.tensor(loss_list, dtype=torch.float).mean()
    acc = num_correct / len(eval_dataloader.dataset)

    writer.add_scalar("val_ranker/loss", epoch_loss, epoch)
    writer.add_scalar("val_ranker/acc", acc, epoch)

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
    criterion = nn.BCEWithLogitsLoss()
    random.seed(config.seed)

    if not os.path.exists(f"weights/{config.experiment_name}"):
        os.makedirs(f"weights/{config.experiment_name}")

    with open("dblp_triplet.pkl", "rb") as f:
        unpickled_data = pickle.load(f)

    # tokenizer = transformers.AutoTokenizer.from_pretrained(config.model_name)
    # embedding_model = EmbeddingModel(config)
    # state_dict = torch.load(
    #     f"weights/{config.experiment_name}/weights_{config.reload_epoch}.pth", 
    #     map_location="cuda:0"
    # )["state_dict"]

    # embedding_model.load_state_dict(state_dict)
    # embedding_model.to(device)

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
    embedding = torch.load("all_embedding_dblp.pth")

    train_query_pair_dataset = QueryPairDataset(unpickled_data["train"], embedding)
    test_query_pair_dataset = QueryPairDataset(unpickled_data["test"], embedding)
        
    train_query_pair_dataloader = DataLoader(
        train_query_pair_dataset,
        batch_size=config.batch_size,
        num_workers=2,
        pin_memory=True,
        shuffle=True
    )
    test_query_pair_dataloader = DataLoader(
        test_query_pair_dataset,
        batch_size=config.batch_size,
        num_workers=2,
        pin_memory=True,
    )

    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer,
        len(train_query_pair_dataloader),
        len(train_query_pair_dataloader) * config.epochs,
    )

    writer = SummaryWriter(f"runs/{config.experiment_name}")

    for epoch in range(0, config.epochs):
        train_one_epoch(
            model,
            train_query_pair_dataloader,
            criterion,
            optimizer,
            scheduler,
            epoch,
            writer,
            device
        )
        eval(model, test_query_pair_dataloader, criterion, epoch, writer, device)

        torch.save(
            {"state_dict": model.state_dict()},
            f"weights/{config.experiment_name}/reranker_weights_{epoch}.pth",
        )
