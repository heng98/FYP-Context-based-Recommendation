import torch
import torch.nn as nn


class SimpleReranker(nn.Module):
    def __init__(self):
        super(SimpleReranker, self).__init__()
        self.word_model = nn.Sequential(
            nn.Linear(300, 50), nn.ReLU(), nn.Dropout(0.3), nn.Linear(50, 1)
        )

        self.linear_1 = nn.Linear(3, 50)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.linear_2 = nn.Linear(50, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, abstract_jaccard, intersection_feature, cos_sim):
        abstract_inter_feat = self.word_model(intersection_feature)

        out = torch.cat([abstract_jaccard, abstract_inter_feat, cos_sim], axis=1)
        out = self.linear_1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.linear_2(out)
        out = self.sigmoid(out)

        return out


class SimpleRerankerForTraining(nn.Module):
    def __init__(self, args):
        super(SimpleRerankerForTraining, self).__init__()
        self.model = SimpleReranker()
        self.criterion = nn.MarginRankingLoss(0.2)

        self.register_buffer(
            'label',
            torch.ones(self.args.batch_size)
        )

    def forward(
        self,
        query_pos_abstract_jaccard,
        query_pos_abstract_inter_feature,
        query_pos_cos_sim,
        query_neg_abstract_jaccard,
        query_neg_abstract_inter_feature,
        query_neg_cos_sim,
    ):
        relevance_pos = self.model(
            query_pos_abstract_jaccard,
            query_pos_abstract_inter_feature,
            query_pos_cos_sim,
        )
        relevance_neg = self.model(
            query_neg_abstract_jaccard,
            query_neg_abstract_inter_feature,
            query_neg_cos_sim
        )
        
        loss = self.criterion(relevance_pos, relevance_neg, self.label)

        return loss

    def save_pretrained(self, path):
        torch.save(self.model.state_dict(), path)

