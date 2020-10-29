import operator
from typing import List, Tuple, Dict, Optional, Generator, NoReturn, Iterator

import math
import numpy as np
import logging
import tqdm
import random

class TripletGenerator():
    def __init__(self, 
                 paper_ids: List[str],
                 dataset: Dict[str, Dict[str, List[int]]],
                 samples_per_query: int,
                 ratio_hard_neg: Optional[float] = 0.5) -> NoReturn:

        self.paper_ids = paper_ids
        self.paper_ids_set = set(paper_ids)
        self.dataset = dataset
        self.samples_per_query = samples_per_query
        self.ratio_hard_neg = ratio_hard_neg
        self.ids2idx = dict()

        for idx, ids in enumerate(self.paper_ids):
            self.ids2idx[ids] = idx

    def get_paper_ids_from_idx(self, idx: int) -> str:
        return self.paper_ids[idx]

    def get_idx_from_paper_ids(self, ids: str) -> int:
        return self.ids2idx[ids]

    def _get_easy_neg(self, query_id: str, n_easy_samples: int) -> List[Tuple[str, str, str]]:
        """Given a query, get easy negative samples
        Easy negative samples are defined as 0 coviews.

        Args:
            query_id: number specifying index of query paper
            n_easy_samples: number of easy samples to output
        """
        not_easy_neg_candidates = self.dataset[query_id]['pos'] + self.dataset[query_id]['hard'] + [query_id]
        easy_neg_candidates = list(self.paper_ids_set.difference(not_easy_neg_candidates))

        easy_samples = []
        if self.dataset[query_id]['pos'] and easy_neg_candidates:
            for i in range(n_easy_samples):
                pos = random.choice(self.dataset[query_id]['pos'])
                neg = random.choice(easy_neg_candidates)
                easy_samples.append((query_id, pos, neg))
        
        return easy_samples

    def _get_hard_neg(self, query_id: str, n_hard_samples: int) -> List[Tuple[str, str, str]]:
        # if there aren't enough candidates to generate enough unique samples
        # reduce the number of samples to make it possible for them to be unique
        n_hard_samples = min(n_hard_samples, len(self.dataset[query_id]['pos'] * len(self.dataset[query_id]['hard'])))

        hard_samples = []
        if self.dataset[query_id]['pos'] and self.dataset[query_id]['hard']:
            for i in range(n_hard_samples):
                pos = random.choice(self.dataset[query_id]['pos'])
                neg = random.choice(self.dataset[query_id]['hard'])
                hard_samples.append((query_id, pos, neg))
            
        return hard_samples


    def _get_triplet(self, query_id: str) -> List[Tuple[str, str, str]]:
        if query_id not in self.paper_ids_set:
            print('Not in')
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

        for query_id in self.dataset:
            results = self._get_triplet(query_id)
            if results:
                for triplet in results:
                    int_triplet = tuple((self.ids2idx[i] for i in triplet))
                    # for i in triplet:
                        # print(self.ids2idx[i])

                    yield int_triplet
                success += 1

            else:
                skipped += 1
             