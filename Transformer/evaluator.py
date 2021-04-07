import torch
from tqdm import tqdm
from collections import defaultdict
import numpy as np
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def mrr(correct):
    try:
        idx = correct.index(True)
        mrr = 1 / (idx + 1)
    except ValueError:
        mrr = 0

    return mrr


def precision_recall_f1(predicted, actual, k=20):
    num_correct = sum(predicted[:k])
    precision = num_correct / k
    recall = num_correct / len(actual)

    if num_correct == 0:
        f1 = 0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1

def average_precision(predicted, actual, k=20):
    predicted_k = predicted[:k]

    score = 0
    num_correct = 0
    for idx, value in enumerate(predicted_k):
        if value:
            num_correct += 1
            score += num_correct / (idx + 1)

    return score / min(len(actual), k)

class Evaluator:
    def __init__(
        self,
        document_embedding_model,
        reranker_model,
        tokenizer,
        ann_candidate_selector,
        candidate_pool,
        device,
    ):
        self.document_embedding_model = document_embedding_model.to(device)
        # self.reranker_model = reranker_model

        self.document_embedding_model.eval()
        # self.reranker_model.eval()

        self.tokenizer = tokenizer

        self.ann_candidate_selector = ann_candidate_selector
        self.candidate_pool = candidate_pool

        self.device = device

    def evaluate(self, test_data):
        skipped = 0
        average_scores = defaultdict(list)
        for i, query in enumerate(tqdm(test_data)):
            pos = set(test_data["pos"]) & self.candidate_pool
            if pos < 10:
                skipped += 1
                continue

            candidates = self.get_candidate(query)

            score = self.eval_score(candidates, pos, [20])
            for k, v in score.items():
                average_scores[k].append(v)

        results = {k: np.mean(v) for k, v in average_scores.items()}

        for k, v in results.items():
            logger.info(f"{k}: {v}")

    def get_candidate(self, query):
        query_embedding = self.get_query_embedding(query)
        candidates = self.ann_candidate_selector.get_candidate(query_embedding)
        candidates = self.rerank(candidates)

        return candidates

    @torch.no_grad()
    def get_query_embedding(self, query):
        encoded_query = self.tokenizer(
            query["title"],
            query["abstract"],
            padding="max_length",
            max_length=256,
            truncation=True,
            return_tensors="pt",
        )

        encoded_query = {k: v.to(self.device) for k, v in encoded_query.items()}
        query_embedding = self.document_embedding_model(**encoded_query)[
            "last_hidden_state"
        ][:, 0]

        return query_embedding.cpu().numpy()[0]

    @torch.no_grad()
    def rerank(self, candidates):
        return candidates

    def eval_score(self, candidates, actual, k_list):
        scores = dict()

        acutal_set = set(actual)
        correct = [y[0] in acutal_set for y in candidates]

        mrr_score = mrr(correct)
        scores["mrr"] = mrr_score

        for k in k_list:
            precision, recall, f1 = precision_recall_f1(correct, actual, k=k)
            ap = average_precision(correct, actual, k=k)
            scores[f"precision@{k}"] = precision
            scores[f"recall@{k}"] = recall
            scores[f"f1@{k}"] = f1
            scores[f"map@{k}"] = ap

        return scores

    def recommend(self, title, abstract):
        recommendation = self.get_candidate(
            {
                "title": title,
                "abstract": abstract
            }
        )
        return recommendation
