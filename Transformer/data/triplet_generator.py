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


EASY_NEG = 1.0
HARD_NEG = 0.6


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
            pos = random.choices(pos_candidates, k=n_easy_samples)
            neg = random.choices(easy_neg_candidates, k=n_easy_samples)

            easy_samples = [(query_id, p, n, EASY_NEG) for p, n in zip(pos, neg)]

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
            pos = random.choices(pos_candidates, k=n_hard_samples)
            neg = random.choices(hard_neg_candidates, k=n_hard_samples)

            hard_samples = [(query_id, p, n, HARD_NEG) for p, n in zip(pos, neg)]

        return hard_samples

    def update_nn_hard(self, doc_embedding):
        logger.info("Updating NN")
        self.nn_neg_flag = True

        distance = pairwise_distances(doc_embedding.numpy())
        top_k = np.argpartition(distance, 10)[:, :10]

        num_of_doc = doc_embedding.shape[0]
        self_idx = np.array(range(num_of_doc)).reshape(num_of_doc, -1)

        top_k = top_k[top_k != self_idx].reshape(num_of_doc, -1)

        # TODO
        for paper_id, top_k_nn in zip(tqdm(self.query_papers_ids), top_k):
            nn_hard_candidate = [self.query_papers_ids[i] for i in top_k_nn]
            data = self.dataset[paper_id]
            nn_hard = list(
                set(nn_hard_candidate) - set(data["hard_neg"]) - set(data["pos"])
            )
            data["nn_neg"] = nn_hard

    def _get_nn_neg(
        self, query_id: str, n_nn_samples: int
    ) -> List[Tuple[str, str, str]]:
        n_nn_samples = min(
            n_nn_samples,
            len(self.dataset[query_id]["pos"] * len(self.dataset[query_id]["nn_neg"])),
        )

        nn_neg_samples = []
        pos_candidates = list(
            set(self.dataset[query_id]["pos"]) & self.candidate_papers_ids
        )
        nn_neg_candidates = list(
            self.candidate_papers_ids & set(self.dataset[query_id]["nn_neg"])
        )

        if pos_candidates and nn_neg_candidates:
            pos = random.choices(pos_candidates, k=n_nn_samples)
            neg = random.choices(nn_neg_candidates, k=n_nn_samples)

            nn_neg_samples = [(query_id, p, n) for p, n in zip(pos, neg)]

        return nn_neg_samples

    def _get_triplet(self, query_id: str) -> List[Tuple[str, str, str]]:
        n_hard_samples = math.ceil(self.ratio_hard_neg * self.samples_per_query)

        if self.nn_neg_flag:
            n_nn_samples = math.ceil(self.ratio_nn_neg * self.samples_per_query)
        else:
            n_nn_samples = 0

        n_easy_samples = self.samples_per_query - n_hard_samples - n_nn_samples

        hard_neg_samples = self._get_hard_neg(query_id, n_hard_samples)
        easy_neg_samples = self._get_easy_neg(query_id, n_easy_samples)

        if self.nn_neg_flag:
            nn_neg_samples = self._get_nn_neg(query_id, n_nn_samples)
        else:
            nn_neg_samples = []

        return hard_neg_samples + easy_neg_samples + nn_neg_samples

    def generate_triplets(self) -> Iterator[Tuple[str, str, str]]:
        """Generate triplets from the whole dataset

        This generates a list of triplets each query according to:
            [(query_id, positive_id, negative_id), ...]

        """
        skipped = 0
        success = 0

        for query_paper_id in self.query_papers_ids:
            results = self._get_triplet(query_paper_id)
            random.shuffle(results)
            if len(results) > 2:
                for triplet in results:
                    yield triplet
                success += 1

            else:
                skipped += 1
