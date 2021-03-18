import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import (
    DataLoader,
    RandomSampler,
    SequentialSampler,
    get_worker_info,
)
from torch.utils.data.distributed import DistributedSampler

import transformers

from model.embedding_model import EmbeddingModel
from model.triplet_loss import TripletLoss
from data.dataset import TripletDataset, TripletCollator, TripletIterableDataset
from utils import distributed
from utils.embed_documents import embed_documents

import os
from tqdm import tqdm
import argparse
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def train_one_epoch(
    model, train_dataloader, criterion, optimizer, scheduler, epoch, writer, config, last_step
):
    logger.info(f"===Training epoch {epoch}===")
    model.train()
    for step, (q, p, n) in enumerate(
        train_dataloader, last_step  # Need check
    ):
        q, p, n = (
            to_device_dict(q, device),
            to_device_dict(p, device),
            to_device_dict(n, device),
        )

        query_embedding = model(q)
        positive_embedding = model(p)
        negative_embedding = model(n)

        loss = criterion(query_embedding, positive_embedding, negative_embedding)
        if config.accumulate_step_size > 1:
            loss = loss / config.accumulate_step_size

        loss.backward()

        if (step + 1) % 10 == 0:
            if config.distributed:
                loss_recorded = distributed.reduce_mean(
                    loss * config.accumulate_step_size
                )
            else:
                loss_recorded = loss.detach().clone()

            if writer:
                writer.add_scalar("embedding/train/loss", loss_recorded.item(), step)

        if (step + 1) % config.accumulate_step_size == 0:
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

    return step


@torch.no_grad()
def eval(model, eval_dataloader, criterion, epoch, writer, config):
    model.eval()

    loss_list = []
    for i, (q, p, n) in enumerate(eval_dataloader, epoch):
        q, p, n = (
            to_device_dict(q, device),
            to_device_dict(p, device),
            to_device_dict(n, device),
        )

        query_embedding = model(q)
        positive_embedding = model(p)
        negative_embedding = model(n)

        loss = criterion(query_embedding, positive_embedding, negative_embedding)

        if config.distributed:
            loss_recorded = distributed.reduce_mean(loss)
        else:
            loss_recorded = loss

        loss_list.append(loss_recorded.item())

    epoch_loss = torch.tensor(loss_list, dtype=torch.float).mean()

    if writer:
        writer.add_scalar("embedding/val/loss", epoch_loss, epoch)


def to_device_dict(d, device):
    return {k: v.to(device) for k, v in d.items()}


def worker_fn(worker_id):
    worker_info = get_worker_info()
    num_workers = worker_info.num_workers
    dataset = worker_info.dataset
    size = len(dataset.query_paper_ids) // num_workers + 1

    dataset.query_paper_ids = dataset.query_paper_ids[
        worker_id * size : (worker_id + 1) * size
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument(
        "--model_name", type=str, default="allenai/scibert_scivocab_cased"
    )
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--accumulate_step_size", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_seq_len", type=int, default=512)

    parser.add_argument("--experiment_name", type=str, required=True)

    parser.add_argument("--samples_per_query", type=int, default=10)
    parser.add_argument("--ratio_hard_neg", type=float, default=0.5)
    parser.add_argument("--ratio_nn_neg", type=float, default=0.1)
    # parser.add_argument("--triplet_dataset_path", type=str, required=True)

    config = parser.parse_args()
    distributed.init_distributed_mode(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = EmbeddingModel(config).to(device)
    criterion = TripletLoss("l2_norm")

    if not os.path.exists(f"weights/{config.experiment_name}"):
        os.makedirs(f"weights/{config.experiment_name}")

    if config.distributed:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[config.gpu])
        config.batch_size //= config.world_size

    with open("./DBLP_train_test_dataset_1.json", "r") as f:
        data_json = json.load(f)
        dataset_name = data_json["name"]
        train_dataset = data_json["train"]
        val_dataset = data_json["valid"]

    train_paper_ids = list(train_dataset.keys())
    val_paper_ids = list(val_dataset.keys())

    dataset = {**train_dataset, **val_dataset}

    train_triplet_dataset = TripletIterableDataset(
        dataset,
        train_paper_ids,
        set(train_paper_ids),
        config
    )
    test_triplet_dataset = TripletIterableDataset(
        dataset,
        val_paper_ids,
        set(train_paper_ids + val_paper_ids),
        config
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(config.model_name)
    collater = TripletCollator(tokenizer, config.max_seq_len)

    # if config.distributed:
    #     train_triplet_sampler = DistributedSampler(train_triplet_dataset, shuffle=True)
    #     test_triplet_sampler = DistributedSampler(test_triplet_dataset, shuffle=False)
    # else:
    #     # Initialize this instead of using shuffle args in DataLoader
    #     train_triplet_sampler = RandomSampler(train_triplet_dataset)
    #     test_triplet_sampler = SequentialSampler(test_triplet_dataset)

    train_triplet_dataloader = DataLoader(
        train_triplet_dataset,
        batch_size=config.batch_size,
        pin_memory=True,
        collate_fn=collater,
        num_workers=8,
        # worker_init_fn=worker_fn
    )
    test_triplet_dataloader = DataLoader(
        test_triplet_dataset,
        batch_size=config.batch_size,
        pin_memory=True,
        collate_fn=collater,
        num_workers=8,
        # worker_init_fn=worker_fn
    )

    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    num_update_steps = (
        len(train_dataset) * config.epochs * config.samples_per_query // config.accumulate_step_size
    )
    scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer,
        num_update_steps * 0.1,
        num_update_steps,
    )

    if distributed.is_main_process():
        writer = SummaryWriter(f"runs/{config.experiment_name}")
    else:
        writer = None

    last_step = 0
    for epoch in range(0, config.epochs):
        # if config.distributed:
        #     train_triplet_sampler.set_epoch(epoch)
        last_step = train_one_epoch(
            model,
            train_triplet_dataloader,
            criterion,
            optimizer,
            scheduler,
            epoch,
            writer,
            config,
            last_step
        )
        eval(model, test_triplet_dataloader, criterion, epoch, writer, config)

        document_embedding = embed_documents(model, train_dataset, tokenizer, device)
        paper_ids_seq = list(train_dataset.keys())
        train_triplet_dataset.triplet_generator.update_nn_hard(document_embedding, paper_ids_seq)

        if distributed.is_main_process():
            torch.save(
                {"state_dict": model.state_dict()},
                f"weights/{config.experiment_name}/weights_{epoch}.pth",
            )
