from typing import List, Tuple, Dict, Optional, Iterator, Set

import math
import random


class TripletGenerator:
    def __init__(
        self,
        query_paper_ids: List[str],
        candidate_papers_ids: Set[str],
        network: Dict[str, Dict[str, List[str]]],
        samples_per_query: int,
        ratio_hard_neg: Optional[float] = 0.5,
    ):
        self.query_papers_ids = query_paper_ids
        self.query_papers_ids_set = set(query_paper_ids)
        self.candidate_papers_ids = candidate_papers_ids
        self.network = network
        self.samples_per_query = samples_per_query
        self.ratio_hard_neg = ratio_hard_neg

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
            self.network[query_id]["pos"] + self.network[query_id]["hard"] + [query_id]
        )
        easy_neg_candidates = list(
            self.candidate_papers_ids.difference(not_easy_neg_candidates)
        )

        easy_samples = []
        pos_candidates = list(set(self.network[query_id]["pos"]) & self.candidate_papers_ids)
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
            len(self.network[query_id]["pos"] * len(self.network[query_id]["hard"])),
        )

        hard_samples = []
        pos_candidates = list(set(self.network[query_id]["pos"]) & self.candidate_papers_ids)
        hard_neg_candidates = list(self.candidate_papers_ids & set(self.network[query_id]["hard"]))

        if pos_candidates and hard_neg_candidates:
            for _ in range(n_hard_samples):
                pos = random.choice(pos_candidates)
                neg = random.choice(hard_neg_candidates)
                hard_samples.append((query_id, pos, neg))

        return hard_samples

    def _get_triplet(self, query_id: str) -> List[Tuple[str, str, str]]:
        if query_id not in self.query_papers_ids_set:
            print("Not in")
            return None

        n_hard_samples = math.ceil(self.ratio_hard_neg * self.samples_per_query)
        n_easy_samples = self.samples_per_query - n_hard_samples

        hard_neg_samples = self._get_hard_neg(query_id, n_hard_samples)
        easy_neg_samples = self._get_easy_neg(query_id, n_easy_samples)

        return hard_neg_samples + easy_neg_samples

    def generate_triplets(self) -> Iterator[Tuple[int, int, int]]:
        """Generate triplets from the whole dataset

        This generates a list of triplets each query according to:
            [(query_id, positive_id, negative_id), ...]

        """
        skipped = 0
        success = 0

        for query_id in self.network:
            results = self._get_triplet(query_id)
            if results:
                for triplet in results:

                    yield triplet
                success += 1

            else:
                skipped += 1

        # print(success, skipped)