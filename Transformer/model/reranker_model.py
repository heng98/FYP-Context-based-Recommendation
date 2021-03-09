import torch
import torch.nn as nn

from transformers import PreTrainedModel, AutoModel


class SimpleReranker(nn.Module):
    def __init__(self):
        super(SimpleReranker, self).__init__()
        self.linear_1 = nn.Linear(2, 50)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.linear_2 = nn.Linear(50, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, similarity, abstract_jaccard):
        out = torch.cat([similarity, abstract_jaccard], axis=1)
        out = self.linear_1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.linear_2(out)
        out = self.sigmoid(out)

        return out
    