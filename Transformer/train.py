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
from uitils import distributed

from tqdm import tqdm
import argparse
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def train_one_epoch(
    model, train_dataloader, criterion, optimizer, scheduler, epoch, writer, config
):
    logger.info()
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
                writer.add_scalars("train/loss", loss_recorded.item(), i)

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
                
                loss_list.append(loss.item())

    epoch_loss = torch.tensor(loss_list, dtype=torch.float).mean()
    writer.add_scalars("val/loss", epoch_loss, epoch)



# def idk():
#         model.eval()
#         with torch.no_grad():
#             doc_embedding_vectors = torch.empty(len(triplet_dataset.paper_ids), 768)
#             for i in tqdm(range(len(triplet_dataset.encoded["input_ids"]), 10)):
#                 q = {k: v[[i]] for k, v in triplet_dataset.encoded.items()}
#                 q = to_device_dict(q, device)
#                 query_embedding = model(q)
#                 doc_embedding_vectors[i] = query_embedding

#             logger.info("Building Annoy Index")
#             ann = ANNAnnoy.build_graph(doc_embedding_vectors)
#             mrr_list = []
#             p_list = []
#             r_list = []
#             f1_list = []

#             logger.info("Evaluating")
#             for query, positive in tqdm(eval_dataset):
#                 if not positive:
#                     continue

#                 query = to_device_dict(query, device)
#                 query_embedding = model(query).cpu().numpy()[0]

#                 # Check if top_k is sorted or not
#                 top_k = ann.get_k_nearest_neighbour(query_embedding, 10)
#                 mrr, precision, recall, f1 = eval_score(top_k, positive)

#                 # logger.info(f"MRR: {mrr}, P@5: {precision}, R@5: {recall}, f1@5: {f1}")
#                 mrr_list.append(mrr)
#                 p_list.append(precision)
#                 r_list.append(recall)
#                 f1_list.append(f1)

#             logger.info("Mean")
#             logger.info(
#                 f"MRR: {sum(mrr_list) / len(mrr_list)}, P@5: {sum(p_list) / len(p_list)}, "
#                 + f"R@5: {sum(r_list) / len(r_list)}, f1@5: {sum(f1_list) / len(f1_list)}"
#             )


def eval_score(predicted_top_k, actual, k=5):
    actual_set = set(actual)
    sorted_correct = [y in actual_set for y in predicted_top_k[0]]

    try:
        idx = sorted_correct.index(True)
        mrr = 1 / (idx + 1)
    except ValueError:
        mrr = 0

    num_correct = sum(sorted_correct[:k])
    precision = num_correct / k
    recall = num_correct / len(actual)

    if num_correct == 0:
        f1 = 0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return mrr, precision, recall, f1


def to_device_dict(d, device):
    return {k: v.to(device) for k, v in d.items()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument(
        "--model_name", type=str, default="allenai/scibert_scivocab_cased"
    )
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--accumulate_step_size", type=int, default=4)

    # parser.add_argument('--local_rank', default=-1, type=int)
    # parser.add_argument('--dataset_path', type=str, required=True)
    # parser.add_argument('--samples_per_query', type=int, default=5)

    config = parser.parse_args()
    distributed.init_distributed_mode(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = EmbeddingModel(config).to(device)
    criterion = TripletLoss()

    if config.distributed:
        model = nn.DistributedDataParallel(model, device_ids=[config.gpu])

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
        test_triplet_sampler = DistributedSampler(test_triplet_dataset)
    else:
        # Initialize this instead of using shuffle args in DataLoader
        train_triplet_sampler = RandomSampler(train_triplet_dataset)
        test_triplet_sampler = SequentialSampler(test_triplet_dataset)

    train_triplet_dataloader = DataLoader(
        train_triplet_dataset,
        batch_size=4,
        num_workers=12,
        pin_memory=True,
        sampler=train_triplet_sampler,
    )
    test_triplet_dataloader = DataLoader(
        test_triplet_dataset,
        batch_size=4,
        num_workers=12,
        pin_memory=True,
        sampler=test_triplet_dataset,
    )

    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer,
        len(train_triplet_dataloader) // 8,
        len(train_triplet_dataloader) * 5 // 8,
    )

    if distributed.is_main_process():
        writer = SummaryWriter()
    else:
        writer = None

    for epoch in range(0, config.epochs):
        train_one_epoch(
            model, train_triplet_dataloader, criterion, optimizer, scheduler, epoch, writer, config
        )
        eval(model, test_triplet_dataloader, criterion, epoch, writer, config)
