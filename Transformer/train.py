import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import transformers

from model.embedding_model import EmbeddingModel
from model.triplet_loss import TripletLoss
from data.dataset import PaperTripletDataset, PaperEvalDataset
from candidate_selector.ann_annoy import ANNAnnoy

from tqdm import tqdm
import argparse
import json
import logging

logging.basicConfig(level = logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# class STLR(torch.optim.lr_scheduler._LRScheduler):
#     def __init__(self, optimizer, max_mul, ratio, steps_per_cycle, decay=1, last_epoch=-1):
#         self.max_mul = max_mul - 1
#         self.turning_point = steps_per_cycle // (ratio + 1)
#         self.steps_per_cycle = steps_per_cycle
#         self.decay = decay
#         super().__init__(optimizer, last_epoch)

#     def get_lr(self):
#         residual = self.last_epoch % self.steps_per_cycle
#         multiplier = self.decay ** (self.last_epoch // self.steps_per_cycle)
#         if residual <= self.turning_point:
#             multiplier *= self.max_mul * (residual / self.turning_point)
#         else:
#             multiplier *= self.max_mul * (
#                 (self.steps_per_cycle - residual) /
#                 (self.steps_per_cycle - self.turning_point))
#         return [lr * (1 + multiplier) for lr in self.base_lrs]


def eval_score(predicted_top_k, actual, k=5):
    actual_set = set(actual)
    sorted_correct = [y in actual_set for y in predicted_top_k[0]]

    try:
        idx = sorted_correct.index(True)
        mrr = 1 / (idx + 1)
    except ValueError:
        mrr = 0

    num_correct = sum(sorted_correct[:k])
    precision = num_correct / k
    recall = num_correct / len(actual)

    if num_correct == 0:
        f1 = 0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return mrr, precision, recall, f1

def to_device_dict(d, device):
    return {k: v.to(device) for k, v in d.items()}

class FullModel(nn.Module):
    def __init__(self, embedding_model, loss_model):
        super(FullModel, self).__init__()
        self.embedding_model = embedding_model
        self.loss_model = loss_model
    
    def forward(self, q, p, n):
        q_e = self.embedding_model(q)
        p_e = self.embedding_model(p)
        n_e = self.embedding_model(n)

        loss = self.loss_model(q_e, p_e, n_e)

        return loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--model_name', type=str, default="allenai/scibert_scivocab_cased")
    parser.add_argument('--lr', type=float, default=2e-4)
    # parser.add_argument('--dataset_path', type=str, required=True)
    # parser.add_argument('--samples_per_query', type=int, default=5)

    config = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    e_model = EmbeddingModel(config)
    criterion = TripletLoss()

    model = FullModel(e_model, criterion).to(device)
    model = nn.DataParallel(model, device_ids=[0, 1, 2, 3])
    
    triplet_dataset = PaperTripletDataset("./train_file.pth")
    eval_dataset = PaperEvalDataset("./test_file.pth", triplet_dataset.idx_paper_ids)

    logger.info(f"{len(triplet_dataset)} triplets is generated")
    triplet_dataloader = DataLoader(triplet_dataset, batch_size=4, shuffle=True, num_workers=12)

    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    scheduler = transformers.get_linear_schedule_with_warmup(optimizer, len(triplet_dataloader) // 8, len(triplet_dataloader) * 5 // 8)

    for epoch in range(config.epochs):
        logger.info(f"=====Epoch {epoch}=====")
        model.train()
        
        for i, (q, p, n) in enumerate(tqdm(triplet_dataloader)):
            q, p, n = to_device_dict(q, device), to_device_dict(p, device), to_device_dict(n, device)
            
            # query_embedding = model(q)
            # positive_embedding = model(p)
            # negative_embedding = model(n)

            # loss = criterion(query_embedding, positive_embedding, negative_embedding)
            loss = model(q, p, n).mean()
            # print(loss)
            loss.backward()

            if (i + 1) % 5000 == 0:
                logger.info(f"LR: {scheduler.get_last_lr()}")

            if (i + 1) % 4 == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

        model.eval()
        with torch.no_grad():
            doc_embedding_vectors = torch.empty(len(triplet_dataset.paper_ids), 768)
            for i in tqdm(range(len(triplet_dataset.encoded["input_ids"]), 10)):
                q = {k: v[[i]] for k, v in triplet_dataset.encoded.items()}
                q = to_device_dict(q, device)
                query_embedding = model(q)
                doc_embedding_vectors[i] = query_embedding

            logger.info("Building Annoy Index")
            ann = ANNAnnoy.build_graph(doc_embedding_vectors)
            mrr_list = []
            p_list = []
            r_list = []
            f1_list = []

            logger.info("Evaluating")
            for query, positive in tqdm(eval_dataset):
                if not positive:
                    continue

                query = to_device_dict(query, device)
                query_embedding = model(query).cpu().numpy()[0]
                
                # Check if top_k is sorted or not
                top_k = ann.get_k_nearest_neighbour(query_embedding, 10)
                mrr, precision, recall, f1 = eval_score(top_k, positive)

                # logger.info(f"MRR: {mrr}, P@5: {precision}, R@5: {recall}, f1@5: {f1}")
                mrr_list.append(mrr)
                p_list.append(precision)
                r_list.append(recall)
                f1_list.append(f1)

            logger.info("Mean")
            logger.info(
                f"MRR: {sum(mrr_list) / len(mrr_list)}, P@5: {sum(p_list) / len(p_list)}, "
                + f"R@5: {sum(r_list) / len(r_list)}, f1@5: {sum(f1_list) / len(f1_list)}"
            )

    # parser = argparse.ArgumentParser()
    # parser.add_argument('--dataset_path', type=str, required=True)
    # parser.add_argument('--samples_per_query', type=int, default=5)

    # args = parser.parse_args()

    # train_set = process_dataset(args.dataset_path)
    # triplet_generator = TripletGenerator(train_set.keys(), train_set, args.samples_per_query)

    # dataset = PaperDataset()
    # model = embedding_model()

    # dataloader = DataLoader(dataset, batch_size=, shuffle=True, drop_last=True)
