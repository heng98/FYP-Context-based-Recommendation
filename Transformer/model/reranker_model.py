import torch
import torch.nn as nn

class SimpleReranker(nn.Module):
    def __init__(self):
        super(SimpleReranker, self).__init__()
        self.linear_1 = nn.Linear(768 * 2, 300)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.linear_2 = nn.Linear(300, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, query_embedding, candidate_embedding):
        mixed_embedding = torch.cat((query_embedding, candidate_embedding), dim=1)

        out = self.linear_1(mixed_embedding)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.linear_2(out)
        out = self.sigmoid(out)

        return out
