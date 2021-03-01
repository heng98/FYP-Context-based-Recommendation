import json

import torch.distributed as dist


import transformers
from transformers import TrainingArguments, HfArgumentParser, TrainerCallback
from transformers.trainer_utils import is_main_process

from model.embedding_model import EmbeddingModel
from argument import ModelArguments, ExperimentArguments
from trainer import TripletTrainer
from data.dataset import TripletCollater, TripletIterableDataset


class SaveModelAtEnd(TrainerCallback):
    def on_epoch_end(self, args: TrainingArguments, state, control, **kwargs):
        control.should_save = True

        return control


if __name__ == "__main__":
    parser = HfArgumentParser((ModelArguments, ExperimentArguments, TrainingArguments))

    model_args, experiment_args, training_args = parser.parse_args_into_dataclasses()

    model = EmbeddingModel(model_args)

    # Initialize data and data processing
    with open(experiment_args.dataset_path, "r") as f:
        data_json = json.load(f)
        dataset_name = data_json["name"]
        train_dataset = data_json["train"]
        val_dataset = data_json["valid"]

    train_paper_ids = list(train_dataset.keys())
    val_paper_ids = list(val_dataset.keys())

    if training_args.local_rank != -1:
        train_split_size = (len(train_paper_ids) // dist.get_world_size()) + 1
        val_split_size = (len(val_paper_ids) // dist.get_world_size()) + 1

        start_split = training_args.local_rank * train_split_size
        end_split = (training_args.local_rank + 1) * train_split_size

        train_query_paper_ids = train_paper_ids[start_split:end_split]

        start_split = training_args.local_rank * val_split_size
        end_split = (training_args.local_rank + 1) * val_split_size

        val_query_paper_ids = val_paper_ids[start_split:end_split]

    else:
        train_query_paper_ids = train_paper_ids[:]
        val_query_paper_ids = val_paper_ids[:]

    train_triplet_dataset = TripletIterableDataset(
        train_dataset, train_paper_ids, set(train_paper_ids), experiment_args
    )
    test_triplet_dataset = TripletIterableDataset(
        val_dataset,
        val_paper_ids,
        set(train_paper_ids + val_paper_ids),
        experiment_args,
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_args.model_name)
    collater = TripletCollater(tokenizer, model_args.max_seq_len)

    trainer = TripletTrainer(
        model=model,
        args=training_args,
        data_collator=collater,
        train_dataset=train_triplet_dataset,
        eval_dataset=test_triplet_dataset,
    )

    trainer.add_callback(SaveModelAtEnd())

    trainer.train()
    