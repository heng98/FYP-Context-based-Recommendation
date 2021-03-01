from dataclasses import dataclass, field

from typing import Optional

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization."
            "Don't set if you want to train a model from scratch."
        },
    )
    max_seq_len: int = field(
        default=256,
    )


@dataclass
class ExperimentArguments:
    """
    Arguments pertaining to what data we are going to do in this training.
    """

    experiment_name: str = field(
        default=None
    )

    dataset_path: str = field(
        default=None
    )

    triplets_per_epoch: int = field(
        default=100000
    )

    samples_per_query: int = field(
        default=15
    )

    ratio_hard_neg: float = field(
        default=0.4
    )

    ratio_nn_neg: float = field(
        default=0.2
    )

