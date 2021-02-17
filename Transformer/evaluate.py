import torch
import numpy as np
from sklearn.metrics import ndcg_score

from torch.utils.data import DataLoader
from model.embedding_model import EmbeddingModel
from model.reranker_model import SimpleReranker
from data.dataset import PaperPosDataset
from candidate_selector.ann.ann_annoy import ANNAnnoy
from candidate_selector.ann.ann_candidate_selector import ANNCandidateSelector

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
    parser.add_argument("--weight_path", type=str, required=True)
    parser.add_argument("--reranker_weight_path", type=str, required=True)
    config = parser.parse_args()

    model = EmbeddingModel(config)
    state_dict = torch.load(config.weight_path, map_location="cuda:1")["state_dict"]
    model.load_state_dict(state_dict)
    # model = AutoModel.from_pretrained(
    #     config.model_name, add_pooling_layer=False, return_dict=True
    # )

    reranker = SimpleReranker()
    reranker_state_dict = torch.load(config.reranker_weight_path)["state_dict"]
    reranker.load_state_dict(reranker_state_dict)

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    reranker = reranker.to(device)
    
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    with open("dblp_train_test_dataset.json", "r") as f:
        dataset = json.load(f)

    train_idx_paper_idx_mapping = {data["ids"]: idx for idx, data in enumerate(dataset["train"])}
    train_paper_pos_dataset = PaperPosDataset(
        dataset["train"],
        train_idx_paper_idx_mapping,
        tokenizer
    )

    test_paper_pos_dataset = PaperPosDataset(
        dataset["test"],
        train_idx_paper_idx_mapping,
        tokenizer
    )

    # train_dataloader = DataLoader(train_paper_pos_dataset, num_workers=8)
    # test_dataloader = DataLoader(test_paper_pos_dataset, num_workers=8)

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

            torch.save(doc_embedding_vectors, "embedding_dblp.pth")

        doc_embedding_vectors = doc_embedding_vectors.cpu().numpy()
        logger.info("Building Annoy Index")
        ann = ANNAnnoy.build_graph(doc_embedding_vectors)
        ann_candidate_selector = ANNCandidateSelector(
            ann, 10, train_paper_pos_dataset, train_idx_paper_idx_mapping
        )
        mrr_list = []
        p_list = []
        r_list = []
        f1_list = []
        ndcg_list = []

        logger.info("Evaluating")
        skipped = 0
        for i, (query, positive) in enumerate(tqdm(test_paper_pos_dataset)):
            if len(positive) < 1:
                skipped += 1
                continue

            query = to_device_dict(query, device)
            query_embedding = model(query)
            query_embedding_numpy = query_embedding.clone().cpu().numpy()[0]

            # Check if top_k is sorted or not
            candidates = ann_candidate_selector.get_candidate(query_embedding_numpy)
            print(candidates)
            mrr_score, precision, recall, f1, ndcg_value = eval_score(candidates, positive, k=20)
            logger.info(f"MRR: {mrr_score}, P@5: {precision}, R@5: {recall}, f1@5: {f1}")
            candidates_ids = [c[0] for c in candidates]

            candidates_vector = torch.from_numpy(
                doc_embedding_vectors[candidates_ids]
            ).to(device)
            output = reranker(
                query_embedding.expand(len(candidates_ids), -1),
                candidates_vector
            ).sigmoid()
            
            # print(output)
            similarity = output.tolist()
            candidates = [(ids, sim) for ids, sim in zip(candidates_ids, similarity)]
            print(candidates)
            candidates = sorted(candidates, key=lambda x: x[1], reverse=True)

            mrr_score, precision, recall, f1, ndcg_value = eval_score(candidates, positive, k=20)

            logger.info(f"MRR: {mrr_score}, P@5: {precision}, R@5: {recall}, f1@5: {f1}")
            mrr_list.append(mrr_score)
            p_list.append(precision)
            r_list.append(recall)
            f1_list.append(f1)
            ndcg_list.append(ndcg_value)

            if i > 10:
                break

        logger.info(f"Skipped: {skipped}")

        logger.info("Mean")
        logger.info(
            f"MRR: {sum(mrr_list) / len(mrr_list)}, P@5: {sum(p_list) / len(p_list)}, "
            + f"R@5: {sum(r_list) / len(r_list)}, f1@5: {sum(f1_list) / len(f1_list)}"
        )
