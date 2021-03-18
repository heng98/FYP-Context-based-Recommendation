from typing import Dict
import torch
import torch.nn as nn

from transformers import AutoModel

from .triplet_loss import TripletLoss

class EmbeddingModel(nn.Module):
    def __init__(self, args):
        super(EmbeddingModel, self).__init__()
        self.args = args
        self.model = AutoModel.from_pretrained(
            args.pretrained_model,
            add_pooling_layer=False,
            return_dict=True,
        )
        self.criterion = TripletLoss("l2_norm")

    def forward(
        self,
        encoded_query: Dict[str, torch.Tensor],
        encoded_positive: Dict[str, torch.Tensor],
        encoded_negative: Dict[str, torch.Tensor],
        margin: torch.Tensor
    ) -> torch.Tensor:
        query_embedding = self.model(**encoded_query)["last_hidden_state"][:, 0]
        positive_embedding = self.model(**encoded_positive)["last_hidden_state"][:, 0]
        negative_embedding = self.model(**encoded_negative)["last_hidden_state"][:, 0]

        loss = self.criterion(query_embedding, positive_embedding, negative_embedding, margin)

        return loss

    def save_pretrained(self, path):
        self.model.save_pretrained(path)
