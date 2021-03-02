import json

import torch.nn as nn
import torch.distributed as dist


import transformers

from model.embedding_model import EmbeddingModel
from utils import distributed
from argument import get_args
from trainer import Trainer
from data.dataset import TripletCollater, TripletIterableDataset


if __name__ == "__main__":
    args = get_args()

    distributed.init_distributed_mode(args)
    model = EmbeddingModel(args)

    # Initialize data and data processing
    with open(args.dataset_path, "r") as f:
        data_json = json.load(f)
        dataset_name = data_json["name"]
        train_dataset = data_json["train"]
        val_dataset = data_json["valid"]

    train_paper_ids = list(train_dataset.keys())
    val_paper_ids = list(val_dataset.keys())

    if args.local_rank != -1:
        train_split_size = (len(train_paper_ids) // dist.get_world_size()) + 1
        val_split_size = (len(val_paper_ids) // dist.get_world_size()) + 1

        start_split = args.local_rank * train_split_size
        end_split = (args.local_rank + 1) * train_split_size

        train_query_paper_ids = train_paper_ids[start_split:end_split]

        start_split = args.local_rank * val_split_size
        end_split = (args.local_rank + 1) * val_split_size

        val_query_paper_ids = val_paper_ids[start_split:end_split]

    else:
        train_query_paper_ids = train_paper_ids[:]
        val_query_paper_ids = val_paper_ids[:]

    train_triplet_dataset = TripletIterableDataset(
        train_dataset, train_paper_ids, set(train_paper_ids), args
    )
    test_triplet_dataset = TripletIterableDataset(
        val_dataset,
        val_paper_ids,
        set(train_paper_ids + val_paper_ids),
        args,
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name)
    collater = TripletCollater(tokenizer, args.max_seq_len)

    model = EmbeddingModel(args)
    if args.distributed:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])

    trainer = Trainer(
        model, train_triplet_dataset, test_triplet_dataset, args, data_collater=collater
    )

    trainer.train()
