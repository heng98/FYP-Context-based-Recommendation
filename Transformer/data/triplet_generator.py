from typing import List, Tuple, Dict, Optional, Iterator, Set, Any

import math
import random
from tqdm import tqdm


class TripletGenerator:
    def __init__(
        self,
        query_paper_ids_idx_mapping: Dict[str, int],
        candidate_papers_ids: Set[str],
        dataset: List[Dict[str, Any]],
        samples_per_query: int,
        ratio_hard_neg: Optional[float] = 0.5,
    ):
        self.query_paper_ids_idx_mapping = query_paper_ids_idx_mapping
        self.candidate_papers_ids = candidate_papers_ids
        self.dataset = dataset

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
        query_idx = self.query_paper_ids_idx_mapping[query_id]
        not_easy_neg_candidates = (
            self.dataset[query_idx]["citations"]
            + self.dataset[query_idx]["hard"]
            + [query_id]
        )
        easy_neg_candidates = list(
            self.candidate_papers_ids.difference(not_easy_neg_candidates)
        )

        easy_samples = []
        pos_candidates = list(
            set(self.dataset[query_idx]["citations"]) & self.candidate_papers_ids
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
        query_idx = self.query_paper_ids_idx_mapping[query_id]
        n_hard_samples = min(
            n_hard_samples,
            len(
                self.dataset[query_idx]["citations"]
                * len(self.dataset[query_idx]["hard"])
            ),
        )

        hard_samples = []
        pos_candidates = list(
            set(self.dataset[query_idx]["citations"]) & self.candidate_papers_ids
        )
        hard_neg_candidates = list(
            self.candidate_papers_ids & set(self.dataset[query_idx]["hard"])
        )

        if pos_candidates and hard_neg_candidates:
            for _ in range(n_hard_samples):
                pos = random.choice(pos_candidates)
                neg = random.choice(hard_neg_candidates)
                hard_samples.append((query_id, pos, neg))

        return hard_samples

    def _get_triplet(self, query_id: str) -> List[Tuple[str, str, str]]:
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

        for data in tqdm(self.dataset):
            if data["ids"] in self.query_paper_ids_idx_mapping:
                results = self._get_triplet(data["ids"])
                for triplet in results:

                    yield triplet
                success += 1

            else:
                skipped += 1


if __name__ == "__main__":
    import pickle
    import json
    from tqdm import tqdm

    with open("aan_dataset.json", "r") as f:
        data_json = json.load(f)
        paper_ids_idx_mapping = data_json["mapping"]
        train_dataset = data_json["train"]
        test_dataset = data_json["test"]

    train_query_paper_ids_idx_mapping = {
        data["ids"]: i for i, data in enumerate(train_dataset)
    }
    test_query_paper_ids_idx_mapping = {
        data["ids"]: i for i, data in enumerate(test_dataset)
    }

    train_triplet_generator = TripletGenerator(
        train_query_paper_ids_idx_mapping,
        set(train_query_paper_ids_idx_mapping.keys()),
        train_dataset,
        5,
    )

    test_candidate_mapping = {
        **train_query_paper_ids_idx_mapping,
        **test_query_paper_ids_idx_mapping,
    }
    test_triplet_generator = TripletGenerator(
        test_query_paper_ids_idx_mapping,
        set(test_candidate_mapping.keys()),
        test_dataset,
        5,
    )

    train_triplet_with_ids = list(
        train_triplet_generator.generate_triplets(),
    )
    test_triplet_with_ids = list(
        test_triplet_generator.generate_triplets(),
    )

    train_triplet = [
        (
            train_query_paper_ids_idx_mapping[q],
            train_query_paper_ids_idx_mapping[p],
            train_query_paper_ids_idx_mapping[n],
        )
        for q, p, n in train_triplet_with_ids
    ]

    test_triplet = [
        (
            test_query_paper_ids_idx_mapping[q],
            test_candidate_mapping[p],
            test_candidate_mapping[n],
        )
        for q, p, n in test_triplet_with_ids
    ]

    with open("aan_triplet.pkl", "wb") as f:
        pickle.dump({"train": train_triplet, "test": test_triplet}, f)
