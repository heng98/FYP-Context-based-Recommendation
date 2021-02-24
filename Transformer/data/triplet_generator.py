from typing import List, Tuple, Dict, Iterator, Set, Any

import math
import numpy as np
from sklearn.metrics import pairwise_distances
import random
from tqdm import tqdm
import argparse
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


class TripletGenerator:
    def __init__(
        self,
        dataset: Dict[str, Dict[str, Any]],
        query_papers_ids: List[str],
        candidate_papers_ids: Set[str],
        config,
    ):
        self.dataset = dataset
        self.query_papers_ids = query_papers_ids
        self.candidate_papers_ids = candidate_papers_ids

        self.samples_per_query = config.samples_per_query
        self.ratio_hard_neg = config.ratio_hard_neg
        self.ratio_nn_neg = config.ratio_nn_neg

        self.nn_neg_flag = False

        assert self.ratio_hard_neg + self.ratio_nn_neg <= 1

    def _get_easy_neg(
        self, query_id: str, n_easy_samples: int
    ) -> List[Tuple[str, str, str]]:
        """Given a query, get easy negative samples
        Easy negative samples are defined as 0 coviews.

        Args:
            query_id: number specifying index of query paper
            n_easy_samples: number of easy samples to output
        """
        not_easy_neg_candidates = (
            self.dataset[query_id]["pos"]
            + self.dataset[query_id]["hard_neg"]
            + self.dataset[query_id].get("nn_neg", [])
            + [query_id]
        )
        easy_neg_candidates = list(
            self.candidate_papers_ids.difference(not_easy_neg_candidates)
        )

        easy_samples = []
        pos_candidates = list(
            (set(self.dataset[query_id]["pos"]) & self.candidate_papers_ids)
            - {query_id}
        )
        if pos_candidates and easy_neg_candidates:
            for _ in range(n_easy_samples):
                pos = random.choice(pos_candidates)
                neg = random.choice(easy_neg_candidates)
                easy_samples.append((query_id, pos, neg))

        return easy_samples

    def _get_hard_neg(
        self, query_id: str, n_hard_samples: int
    ) -> List[Tuple[str, str, str]]:
        # if there aren't enough candidates to generate enough unique samples
        # reduce the number of samples to make it possible for them to be unique
        n_hard_samples = min(
            n_hard_samples,
            len(
                self.dataset[query_id]["pos"] * len(self.dataset[query_id]["hard_neg"])
            ),
        )

        hard_samples = []
        pos_candidates = list(
            (set(self.dataset[query_id]["pos"]) & self.candidate_papers_ids)
            - {query_id}
        )
        hard_neg_candidates = list(
            (self.candidate_papers_ids & set(self.dataset[query_id]["hard_neg"]))
            - {query_id}
        )

        if pos_candidates and hard_neg_candidates:
            for _ in range(n_hard_samples):
                pos = random.choice(pos_candidates)
                neg = random.choice(hard_neg_candidates)
                hard_samples.append((query_id, pos, neg))

        return hard_samples

    def update_nn_hard(self, doc_embedding, paper_ids_seq):
        logger.info("Updating NN")
        self.nn_neg_flag = True

        distance = pairwise_distances(doc_embedding)
        top_k = np.argpartition(distance, 10)[:10]

        num_of_doc = doc_embedding.shape[0]
        self_idx = np.array(range(num_of_doc)).reshape(num_of_doc, -1)

        top_k = top_k[top_k != self_idx].reshape(num_of_doc, -1)

        for paper_id, top_k_nn in zip(tqdm(paper_ids_seq), top_k):
            nn_hard_candidate = [paper_ids_seq[i] for i in top_k_nn]
            data = self.dataset[paper_id]
            nn_hard = list(
                set(nn_hard_candidate) - set(data["hard_neg"]) - set(data["pos"])
            )
            data["nn_neg"] = nn_hard

    def _get_nn_neg(
        self, query_id: str, n_hard_samples: int
    ) -> List[Tuple[str, str, str]]:
        n_hard_samples = min(
            n_hard_samples,
            len(self.dataset[query_id]["pos"] * len(self.dataset[query_id]["nn_neg"])),
        )

        nn_samples = []
        pos_candidates = list(
            set(self.dataset[query_id]["pos"]) & self.candidate_papers_ids
        )
        nn_neg_candidates = list(
            self.candidate_papers_ids & set(self.dataset[query_id]["nn_neg"])
        )

        if pos_candidates and nn_neg_candidates:
            for _ in range(n_hard_samples):
                pos = random.choice(pos_candidates)
                neg = random.choice(nn_neg_candidates)
                nn_samples.append((query_id, pos, neg))

        return nn_samples

    def _get_triplet(self, query_id: str) -> List[Tuple[str, str, str]]:
        n_hard_samples = math.ceil(self.ratio_hard_neg * self.samples_per_query)

        if self.nn_neg_flag:
            n_nn_samples = math.ceil(self.ratio_nn_neg * self.samples_per_query)
        else:
            n_nn_samples = 0

        n_easy_samples = self.samples_per_query - n_hard_samples - n_nn_samples

        hard_neg_samples = self._get_hard_neg(query_id, n_hard_samples)
        easy_neg_samples = self._get_easy_neg(query_id, n_easy_samples)

        return hard_neg_samples + easy_neg_samples

    def generate_triplets(self) -> Iterator[Tuple[str, str, str]]:
        """Generate triplets from the whole dataset

        This generates a list of triplets each query according to:
            [(query_id, positive_id, negative_id), ...]

        """
        skipped = 0
        success = 0

        for query_paper_id in tqdm(self.query_papers_ids):
            results = self._get_triplet(query_paper_id)
            if len(results) > 2:
                for triplet in results:
                    yield triplet
                success += 1

            else:
                skipped += 1


if __name__ == "__main__":
    exit()
    import pickle
    import json
    from multiprocessing_generator import ParallelGenerator

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_test_dataset", type=str, required=True)
    config = parser.parse_args()

    with open(config.train_test_dataset, "r") as f:
        data_json = json.load(f)
        dataset_name = data_json["name"]
        train_dataset = data_json["train"]
        val_dataset = data_json["valid"]

    all_query_paper_ids_idx_mapping = {
        data["ids"]: i for i, data in enumerate(train_dataset + val_dataset)
    }
    train_paper_ids_idx_mapping = {
        data["ids"]: i for i, data in enumerate(train_dataset)
    }
    val_paper_ids_idx_mapping = {data["ids"]: i for i, data in enumerate(val_dataset)}

    train_candidate = set(train_paper_ids_idx_mapping.keys())
    val_candidate = set(all_query_paper_ids_idx_mapping.keys())

    train_triplet_generator = TripletGenerator(
        train_paper_ids_idx_mapping,
        train_candidate,
        train_dataset,
        10,
    )

    val_triplet_generator = TripletGenerator(
        val_paper_ids_idx_mapping,
        val_candidate,
        val_dataset,
        10,
    )

    with ParallelGenerator(
        train_triplet_generator.generate_triplets(),
        max_lookahead=100,
    ) as g:

        train_triplet_with_ids = list(tqdm(g))

    with ParallelGenerator(
        val_triplet_generator.generate_triplets(), max_lookahead=100
    ) as g:
        val_triplet_with_ids = list(tqdm(g))

    train_triplet = [
        (
            all_query_paper_ids_idx_mapping[q],
            all_query_paper_ids_idx_mapping[p],
            all_query_paper_ids_idx_mapping[n],
        )
        for q, p, n in train_triplet_with_ids
    ]

    val_triplet = [
        (
            all_query_paper_ids_idx_mapping[q],
            all_query_paper_ids_idx_mapping[p],
            all_query_paper_ids_idx_mapping[n],
        )
        for q, p, n in val_triplet_with_ids
    ]

    train_triplet = list(set(train_triplet))
    val_triplet = list(set(val_triplet))

    logger.info(f"No of train triplet: {len(train_triplet)}")
    logger.info(f"No of validation triplet: {len(val_triplet)}")

    with open(f"{dataset_name}_10_triplet.pkl", "wb") as f:
        pickle.dump(
            {
                "dataset": train_dataset + val_dataset,
                "train": train_triplet,
                "valid": val_triplet,
            },
            f,
        )
