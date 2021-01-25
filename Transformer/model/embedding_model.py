from typing import Dict
import torch
import torch.nn as nn

from transformers import AutoModel


class EmbeddingModel(nn.Module):
    def __init__(self, config):
        super(EmbeddingModel, self).__init__()
        self.config = config
        self.model = AutoModel.from_pretrained(
            self.config.model_name, add_pooling_layer=False, return_dict=True
        )

    def forward(self, input: Dict[str, torch.Tensor]) -> torch.Tensor:
        output = self.model(
            input_ids=input["input_ids"],
            attention_mask=input["attention_mask"],
            token_type_ids=input["token_type_ids"],
            position_ids=torch.arange(512).expand((1, -1)).cuda(),
        )
        doc_embedding = output["last_hidden_state"][:, 0]

        return doc_embedding
