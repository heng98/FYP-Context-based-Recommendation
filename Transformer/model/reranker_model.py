import torch.nn as nn

class SimpleReranker:
    def __init__(self):
        super(SimpleReranker, self).__init__()
        self.linear_1 = nn.Linear(768, 300)
        self.relu = nn.ReLU()
        self.linear_2 = nn.Linear(300, 1)

    def forward(self, query_embedding, candidate_embedding):
        mixed_embedding = torch.add(query_embedding, candidate_embedding)

        out = self.linear_1(mixed_embedding)
        out = self.relu(out)
        out = self.linear_2(out)

        return out
