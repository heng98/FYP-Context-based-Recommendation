import json
import datasets
import pickle

import torch.nn as nn
import torch.distributed as dist

from transformers import AutoTokenizer, AutoModel

from model.embedding_model import EmbeddingModel
from utils import distributed
from argument import get_args
from trainer import Trainer
from data.dataset import TripletCollater, TripletIterableDataset
from data.corpus import S2ORCCorpus


if __name__ == "__main__":
    args = get_args()

    distributed.init_distributed_mode(args)
    
    # Initialize data and data processing
    # with open(args.dataset_path, "r") as f:
    #     data_json = json.load(f)
    #     dataset_name = data_json["name"]
    #     train_dataset = data_json["train"]
    #     val_dataset = data_json["valid"]

    

    # dataset = {**train_dataset, **val_dataset}

    # train_paper_ids = list(train_dataset.keys())
    # val_paper_ids = list(val_dataset.keys())
    with open("Dataset/processed/s2orc_cs/train_ids.pkl", "rb") as g:
        meta = pickle.load(g)

    hf_dataset = datasets.load_dataset(
        'json',
        name="cs_paper",
        data_files=["Dataset/processed/s2orc_cs/s2orc_train.json", "Dataset/processed/s2orc_cs/s2orc_val.json"],
        split='train'
    )
    dataset = S2ORCCorpus(hf_dataset, meta["paper_ids_idx_mapping"])
    train_paper_ids = meta["train_ids"]
    val_paper_ids = meta["val_ids"]


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
        dataset, train_paper_ids, set(train_paper_ids), args.train_triplets_per_epoch, args
    )
    test_triplet_dataset = TripletIterableDataset(
        dataset,
        val_paper_ids,
        set(train_paper_ids + val_paper_ids),
        args.eval_triplets_per_epoch,
        args
    )

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)
    collater = TripletCollater(tokenizer, args.max_seq_len)
    
    model = EmbeddingModel(args)

    trainer = Trainer(
        model, train_triplet_dataset, test_triplet_dataset, args, data_collater=collater
    )

    trainer.train()
