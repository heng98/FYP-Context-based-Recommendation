import torch
import numpy as np
from sklearn.metrics import ndcg_score

from torch.utils.data import DataLoader
from model.embedding_model import EmbeddingModel
from model.reranker_model import SimpleReranker
from data.dataset import PaperPosDataset
from candidate_selector.ann.ann_annoy import ANNAnnoy
from candidate_selector.ann.ann_candidate_selector import ANNCandidateSelector
from ranker import Ranker

import argparse
import json
from tqdm import tqdm
import logging
from transformers import AutoTokenizer, AutoModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def to_device_dict(d, device):
    return {k: v.to(device) for k, v in d.items()}

def eval_score(predicted, actual, k=20):
    actual_set = set(actual)
    correct = [y[0] in actual_set for y in predicted]

    mrr_score = mrr(correct, k=k)
    precision, recall, f1 = precision_recall_f1(correct, actual, k=k)
    ndcg_value = ndcg(predicted, actual, k=k)

    return mrr_score, precision, recall, f1, ndcg_value

def mrr(correct, k=20):
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


def ndcg(predicted, actual, k=20):
    return 0
    actual_set = set(actual)
    sorted_correct = [1 if y in actual_set else 0 for y in predicted[0]]
    print(sorted_correct[:k], predicted[:k])
    score = ndcg_score(sorted_correct[:k], predicted[0][:k])
    
    return score


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", type=str, default="allenai/scibert_scivocab_cased"
    )
    parser.add_argument("--embedding_path", type=str)
    parser.add_argument("--max_seq_len", type=int, default=256)
    parser.add_argument("--weight_path", type=str, required=True)


    parser.add_argument("--reranker_weight_path", type=str)
    config = parser.parse_args()

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    model = EmbeddingModel(config)
    state_dict = torch.load(config.weight_path, map_location=device)["state_dict"]
    model.load_state_dict(state_dict)
    model = model.to(device)
    
    if config.reranker_weight_path:
        reranker_model = SimpleReranker()
        reranker_state_dict = torch.load(config.reranker_weight_path)["state_dict"]
        reranker_model.load_state_dict(reranker_state_dict)
        reranker_model = reranker_model.to(device)
    
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    with open("DBLP_train_test_dataset_1.json", "r") as f:
        dataset = json.load(f)

    train_idx_paper_idx_mapping = {ids: idx for idx, ids in enumerate(dataset["train"].keys())}
    train_paper_pos_dataset = PaperPosDataset(
        list(dataset["train"].values()),
        train_idx_paper_idx_mapping,
        tokenizer
    )

    test_paper_pos_dataset = PaperPosDataset(
        list(dataset["test"].values()),
        train_idx_paper_idx_mapping,
        tokenizer,
        abstract=False
    )

    model.eval()
    with torch.no_grad():
        if config.embedding_path:
            logger.info("Load from embedding")
            doc_embedding_vectors = torch.load(config.embedding_path)
        else:
            doc_embedding_vectors = torch.empty(len(train_paper_pos_dataset), 768)
            for i, (encoded, _) in enumerate(tqdm(train_paper_pos_dataset)):
                encoded = to_device_dict(encoded, device)
        
                query_embedding = model(encoded)
                doc_embedding_vectors[i] = query_embedding

            torch.save(doc_embedding_vectors, "embedding_dblp_2.pth")

        doc_embedding_vectors = doc_embedding_vectors.cpu().numpy()
        logger.info("Building Annoy Index")
        ann = ANNAnnoy.build_graph(doc_embedding_vectors)
        # 8
        ann_candidate_selector = ANNCandidateSelector(
            ann, 8, train_paper_pos_dataset, train_idx_paper_idx_mapping
        )
        if config.reranker_weight_path:
            ranker = Ranker(reranker_model, doc_embedding_vectors, device)

        mrr_list = []
        p_list = []
        r_list = []
        f1_list = []
        ndcg_list = []

        logger.info("Evaluating")
        skipped = 0
        for i, (query, positive) in enumerate(tqdm(test_paper_pos_dataset)):
            if len(positive) < 10:
                skipped += 1
                continue

            query = to_device_dict(query, device)
            query_embedding = model(query)
            query_embedding_numpy = query_embedding.clone().cpu().numpy()[0]

            candidates = ann_candidate_selector.get_candidate(query_embedding_numpy)
           
            if config.reranker_weight_path:
                candidates = ranker.rank(query_embedding, candidates)

            # logger.info(dataset["test"][i])
            # for c in candidates:
            #     logger.info(dataset["train"][c[0]]["title"])
            

            mrr_score, precision, recall, f1, ndcg_value = eval_score(candidates, positive, k=20)

            logger.info(f"MRR: {mrr_score}, P@5: {precision}, R@5: {recall}, f1@5: {f1}")
            mrr_list.append(mrr_score)
            p_list.append(precision)
            r_list.append(recall)
            f1_list.append(f1)
            ndcg_list.append(ndcg_value)
            # break

        logger.info(f"Skipped: {skipped}")

        logger.info("Mean")
        logger.info(
            f"MRR: {sum(mrr_list) / len(mrr_list)}, P@5: {sum(p_list) / len(p_list)}, "
            + f"R@5: {sum(r_list) / len(r_list)}, f1@5: {sum(f1_list) / len(f1_list)}"
        )
        logger.info(
            f"MRR: {np.mean(mrr_list)}, P@5: {np.mean(p_list)}, "
            + f"R@5: {np.mean(r_list)}, f1@5: {np.mean(f1_list)}"
        )
