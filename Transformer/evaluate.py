import torch
import numpy as np
from sklearn.metrics import ndcg_score

from model.embedding_model import EmbeddingModel
from data.dataset import PaperDataset
from candidate_selector.ann_annoy import ANNAnnoy

import argparse
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def to_device_dict(d, device):
    return {k: v.to(device) for k, v in d.items()}

def eval_score(predicted, actual, k=20):
    actual_set = set(actual)
    correct = [y in actual_set for y in predicted]


    mrr_score = mrr(correct, k=k)
    precision, recall, f1 = precision_recall_f1(correct, actual, k=k)
    ndcg_value = ndcg(predicted, actual, k=k)

    return mrr_score, precision, recall, f1, ndcg_value

def mrr(predicted, actual, k=20):
    return 0
    try:
        idx = sorted_correct.index(True)
        mrr = 1 / (idx + 1)
    except ValueError:
        mrr = 0

    return mrr


def precision_recall_f1(predicted, actual, k=20):
    num_correct = sum(predicted)
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
    parser.add_argument("--weight_path", type=str, required=True)
    config = parser.parse_args()

    model = EmbeddingModel(config)
    state_dict = torch.load(config.weight_path)["state_dict"]
    temp_fix_state_dict = {}
    for k, v in state_dict.items():
        temp_fix_state_dict[k.replace("module.", "")] = v


    model.load_state_dict(temp_fix_state_dict)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    train_paper_dataset = PaperDataset("./train_file.pth", "./encoded.pth", config)
    train_paper_pos_dataset = train_paper_dataset.get_paper_pos_dataset(
        train_paper_dataset.paper_ids_idx_mapping
    )

    test_paper_dataset = PaperDataset("./test_file.pth", "./encoded.pth", config)
    test_paper_pos_dataset = test_paper_dataset.get_paper_pos_dataset(
        train_paper_dataset.paper_ids_idx_mapping
    )

    model.eval()
    with torch.no_grad():
        doc_embedding_vectors = torch.empty(len(train_paper_pos_dataset), 768)
        for i, (encoded, _) in enumerate(tqdm(train_paper_pos_dataset)):
            encoded = {k: v.unsqueeze(0) for k, v in encoded.items()}
            encoded = to_device_dict(encoded, device)
    
            query_embedding = model(encoded)
            doc_embedding_vectors[i] = query_embedding

        doc_embedding_vectors = doc_embedding_vectors.cpu().numpy()
        logger.info("Building Annoy Index")
        ann = ANNAnnoy.build_graph(doc_embedding_vectors)
        mrr_list = []
        p_list = []
        r_list = []
        f1_list = []
        ndcg_list = []

        logger.info("Evaluating")
        skipped = 0
        for i, (query, positive) in enumerate(tqdm(train_paper_pos_dataset)):
            if not positive:
                skipped += 1
                continue

            query = {k: v.unsqueeze(0) for k, v in query.items()}
            query = to_device_dict(query, device)
            query_embedding = model(query).cpu().numpy()[0]

            # Check if top_k is sorted or not
            top_k = ann.get_k_nearest_neighbour(query_embedding, 5)
            candidate = set(top_k[0])
            for idx in top_k[0]:
                _, citation_of_top_k = train_paper_dataset[idx]
                candidate += set(citation_of_top_k)

            mrr_score, precision, recall, f1, ndcg_value = eval_score(candidate, positive, k=10)

            logger.info(f"MRR: {mrr_score}, P@5: {precision}, R@5: {recall}, f1@5: {f1}")
            mrr_list.append(mrr_score)
            p_list.append(precision)
            r_list.append(recall)
            f1_list.append(f1)
            ndcg_list.append(ndcg_value)

        logger.info(f"Skipped: {skipped}")

        logger.info("Mean")
        logger.info(
            f"MRR: {sum(mrr_list) / len(mrr_list)}, P@5: {sum(p_list) / len(p_list)}, "
            + f"R@5: {sum(r_list) / len(r_list)}, f1@5: {sum(f1_list) / len(f1_list)}"
        )
