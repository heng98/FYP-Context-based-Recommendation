import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

import transformers

from model.embedding_model import EmbeddingModel
from model.triplet_loss import TripletLoss
from data.dataset import PaperDataset
from candidate_selector.ann_annoy import ANNAnnoy
from utils import distributed

from tqdm import tqdm
import argparse
import random
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def train_one_epoch(
    model, train_dataloader, criterion, optimizer, scheduler, epoch, writer, config
):
    logger.info(f"===Training epoch {epoch}===")
    model.train()
    for i, (q, p, n) in enumerate(
        tqdm(train_dataloader), epoch * len(train_dataloader)       # Need check
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

        if i % 50 == 0:
            if config.distributed:
                loss_recorded = distributed.reduce_mean(loss)
            else:
                loss_recorded = loss.clone().detach()

            if writer:
                writer.add_scalar("train/loss", loss_recorded.item(), i)

        loss.backward()

        if (i + 1) % 4 == 0:
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()


def eval(model, eval_dataloader, criterion, epoch, writer, config):
    model.eval()

    loss_list = []
    with torch.no_grad():
        for i, (q, p, n) in enumerate(tqdm(eval_dataloader), epoch):
            q, p, n = (
                to_device_dict(q, device),
                to_device_dict(p, device),
                to_device_dict(n, device),
            )

            query_embedding = model(q)
            positive_embedding = model(p)
            negative_embedding = model(n)

            loss = criterion(query_embedding, positive_embedding, negative_embedding)
    
            if i % 50 == 0:
                if config.distributed:
                    loss_recorded = distributed.reduce_mean(loss)
                else:
                    loss_recorded = loss
                
                loss_list.append(loss_recorded.item())

    epoch_loss = torch.tensor(loss_list, dtype=torch.float).mean()

    if writer:
        writer.add_scalar("val/loss", epoch_loss, epoch)

def to_device_dict(d, device):
    return {k: v.to(device) for k, v in d.items()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument(
        "--model_name", type=str, default="allenai/scibert_scivocab_cased"
    )
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--accumulate_step_size", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=8)

    parser.add_argument('--samples_per_query', type=int, default=5)
    parser.add_argument('--ratio_hard_neg', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=42)

    config = parser.parse_args()
    distributed.init_distributed_mode(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = EmbeddingModel(config).to(device)
    criterion = TripletLoss()
    
    random.seed(config.seed)

    if config.distributed:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[config.gpu])
        config.batch_size //= config.world_size

    train_paper_dataset = PaperDataset("./train_file.pth", "./encoded.pth", config)
    train_triplet_dataset = train_paper_dataset.get_triplet_dataset(
        train_paper_dataset.paper_ids_idx_mapping
    )

    test_paper_dataset = PaperDataset("./test_file.pth", "./encoded.pth", config)
    test_triplet_dataset = test_paper_dataset.get_triplet_dataset(
        {
            **train_paper_dataset.paper_ids_idx_mapping,
            **test_paper_dataset.paper_ids_idx_mapping,
        }
    )
    test_paper_pos_dataset = test_paper_dataset.get_paper_pos_dataset(
        train_paper_dataset.paper_ids_idx_mapping
    )

    if config.distributed:
        train_triplet_sampler = DistributedSampler(train_triplet_dataset)
        test_triplet_sampler = DistributedSampler(test_triplet_dataset, shuffle=False)
    else:
        # Initialize this instead of using shuffle args in DataLoader
        train_triplet_sampler = RandomSampler(train_triplet_dataset)
        test_triplet_sampler = SequentialSampler(test_triplet_dataset)

    train_triplet_dataloader = DataLoader(
        train_triplet_dataset,
        batch_size=config.batch_size,
        num_workers=12,
        pin_memory=True,
        sampler=train_triplet_sampler,
    )
    test_triplet_dataloader = DataLoader(
        test_triplet_dataset,
        batch_size=config.batch_size,
        num_workers=12,
        pin_memory=True,
        sampler=test_triplet_sampler,
    )

    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer,
        len(train_triplet_dataloader),
        len(train_triplet_dataloader) * config.epochs,
    )

    if distributed.is_main_process():
        writer = SummaryWriter()
    else:
        writer = None

    for epoch in range(0, config.epochs):
        if config.distributed:
            train_triplet_sampler.set_epoch(epoch)
        train_one_epoch(
            model, train_triplet_dataloader, criterion, optimizer, scheduler, epoch, writer, config
        )
        eval(model, test_triplet_dataloader, criterion, epoch, writer, config)


        if distributed.is_main_process():
            torch.save({'state_dict': model.state_dict()}, f'weights_{epoch}.pth')