import torch
import torch.distributed as dist

from data.dataset import TripletRankerCollater, TripletIterableDataset
from data.preprocessor import SimplerRankerPreprocessor
from model.reranker_model import SimpleRerankerForTraining
from utils import distributed
from argument import get_args
from trainer import Trainer

import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


if __name__ == "__main__":
    args = get_args()

    distributed.init_distributed_mode(args)

    # Initialize data and data processing
    with open(args.dataset_path, "r") as f:
        data_json = json.load(f)
        dataset_name = data_json["name"]
        train_dataset = data_json["train"]
        val_dataset = data_json["valid"]

    dataset = {**train_dataset, **val_dataset}
    paper_ids_idx_mapping = {ids: idx for idx, ids in enumerate(train_dataset)}

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

    doc_embedding= torch.load("")
    preprocessor = SimplerRankerPreprocessor(
        doc_embedding, paper_ids_idx_mapping, ""
    )


    train_triplet_dataset = TripletIterableDataset(
        dataset,
        train_paper_ids,
        set(train_paper_ids),
        args.train_triplets_per_epoch,
        args,
        preprocessor=preprocessor
    )
    train_triplet_dataset.triplet_generator.update_nn_hard(doc_embedding)
    test_triplet_dataset = TripletIterableDataset(
        dataset,
        [],
        # val_paper_ids,
        set(train_paper_ids + val_paper_ids),
        args.eval_triplets_per_epoch,
        args,
        preprocessor=preprocessor
    )

    model = SimpleRerankerForTraining(args)

    trainer = Trainer(
        model, train_triplet_dataset, test_triplet_dataset, args
    )

    trainer.train()
