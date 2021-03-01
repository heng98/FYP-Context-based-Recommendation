from typing import Dict
import torch
import torch.nn as nn

from transformers import AutoModel, PreTrainedModel


class EmbeddingModel(PreTrainedModel):
    def __init__(self, model_args):
        super(EmbeddingModel, self).__init__()
        self.model_args = model_args
        self.model = AutoModel.from_pretrained(
            self.model_args.model_name_or_path,
            add_pooling_layer=False,
            return_dict=True,
        )

        self.register_buffer(
            "position_ids", torch.arange(model_args.max_seq_len).expand((1, -1))
        )

    def forward(self, input: Dict[str, torch.Tensor]) -> torch.Tensor:
        output = self.model(
            input_ids=input["input_ids"],
            attention_mask=input["attention_mask"],
            token_type_ids=input["token_type_ids"],
            position_ids=self.position_ids.clone(),
        )
        doc_embedding = output["last_hidden_state"][:, 0]

        return doc_embedding
