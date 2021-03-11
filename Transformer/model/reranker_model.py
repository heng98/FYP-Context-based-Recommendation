import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModelForSequenceClassification

class SimpleReranker(nn.Module):
    def __init__(self):
        super(SimpleReranker, self).__init__()
        self.projection = nn.Sequential(
            nn.Linear(300, 300),
        )

        self.word_model = nn.Sequential(
            nn.Linear(300, 50), nn.ReLU(), nn.Dropout(0.3), nn.Linear(50, 1)
        )

        self.relevance_model = nn.Sequential(
            nn.Linear(5, 20),
            nn.ELU(),
            nn.Linear(20, 20),
            nn.ELU(),
            nn.Linear(20, 1),
            nn.Sigmoid(),
        )

    def forward(
            self, 
            query_title_embedding,
            candidate_title_embedding,
            query_abstract_embedding,
            candidate_abstract_embedding,
            abstract_jaccard, 
            intersection_feature, 
            cos_sim
        ):
        query_title_proj = self.projection(query_title_embedding)
        candidate_title_proj = self.projection(candidate_title_embedding)

        query_abstract_proj = self.projection(query_abstract_embedding)
        candidate_abstract_proj = self.projection(candidate_abstract_embedding)

        title_cos_sim = F.cosine_similarity(query_title_proj, candidate_title_proj).unsqueeze(1)
        abstract_cos_sim = F.cosine_similarity(query_abstract_proj, candidate_abstract_proj).unsqueeze(1)
        abstract_inter_feat = self.word_model(intersection_feature)

        out = torch.cat([title_cos_sim, abstract_cos_sim, abstract_jaccard, abstract_inter_feat, cos_sim], axis=1)
        out = self.relevance_model(out)

        return out


class SimpleRerankerForTraining(nn.Module):
    def __init__(self, args):
        super(SimpleRerankerForTraining, self).__init__()
        self.model = SimpleReranker()
        # self.criterion = nn.MarginRankingLoss(0.1)
        self.args = args

        # self.register_buffer(
        #     'label',
        #     torch.ones(self.args.batch_size)
        # )

    def forward(
        self,
        query_title_embedding,
        pos_title_embedding,
        neg_title_embedding,
        query_abstract_embedding,
        pos_abstract_embedding,
        neg_abstract_embedding,
        query_pos_abstract_jaccard,
        query_pos_abstract_inter_feature,
        query_pos_cos_sim,
        query_neg_abstract_jaccard,
        query_neg_abstract_inter_feature,
        query_neg_cos_sim,
    ):
        relevance_pos = self.model(
            query_title_embedding,
            pos_title_embedding,
            query_abstract_embedding,
            pos_abstract_embedding,
            query_pos_abstract_jaccard,
            query_pos_abstract_inter_feature,
            query_pos_cos_sim,
        )
        relevance_neg = self.model(
            query_title_embedding,
            neg_title_embedding,
            query_abstract_embedding,
            neg_abstract_embedding,
            query_neg_abstract_jaccard,
            query_neg_abstract_inter_feature,
            query_neg_cos_sim
        )
        # print(relevance_pos)
        # loss = self.criterion(relevance_pos, relevance_neg, self.label)
        loss = F.relu(relevance_neg - relevance_pos + 0.1).mean()

        return loss

    def save_pretrained(self, path):
        torch.save(self.model.state_dict(), path)


class TransformerRanker(nn.Module):
    def __init__(self, args):
        super(TransformerRanker, self).__init__()
        self.args = args

        self.model = AutoModelForSequenceClassification(
            args.pretrained_model,
            return_dict=True,
            num_labels=1
        )

        self.criterion = nn.CrossEntropyLoss()
        self.register_buffer(
            "label",
            torch.zeros(self.args.batch_size, dtype=torch.long)
        )

    def forward(self, inputs):
        logits = self.model(**inputs)["logits"]

        scores = logits.view(
            self.args.batch_size,
            self.args.train_group_size
        )

        loss = self.criterion(scores, self.label)

        return loss