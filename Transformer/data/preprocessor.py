from typing import List

from torch.nn.functional import cosine_similarity
from torchtext.data.utils import get_tokenizer

import fasttext
import numpy as np


class Preprocessor:
    def __call__(self, query_paper, pos_paper, neg_paper):
        raise NotImplementedError


class DefaultPreprocessor(Preprocessor):
    def __call__(self, query_paper, pos_paper, neg_paper):
        return {
            "query_paper": query_paper,
            "pos_paper": pos_paper,
            "neg_paper": neg_paper,
        }


class SimplerRankerPreprocessor(Preprocessor):
    def __init__(self, doc_embedding, paper_ids_idx_mapping, fasttext_path):
        self.doc_embedding = doc_embedding
        self.paper_ids_idx_mapping = paper_ids_idx_mapping

        self.model = fasttext.load_model(fasttext_path)
        self.tokenizer = get_tokenizer("basic_english")

    def __call__(self, query_paper, pos_paper, neg_paper):
        tokenized_query_abstract = self.tokenizer(query_paper["abstract"])
        tokenized_pos_abstract = self.tokenizer(pos_paper["abstract"])
        tokenized_neg_abstract = self.tokenizer(neg_paper["abstract"])

        # Jaccard
        query_pos_abstract_jaccard = self._jaccard(
            tokenized_query_abstract, tokenized_pos_abstract
        )
        query_neg_abstract_jaccard = self._jaccard(
            tokenized_query_abstract, tokenized_neg_abstract
        )

        # Intersection feature
        query_pos_abstract_inter_feature = self._intersection_feature(
            tokenized_query_abstract, tokenized_pos_abstract
        )
        query_neg_abstract_inter_feature = self._intersection_feature(
            tokenized_query_abstract, tokenized_neg_abstract
        )

        # Cosine Similarity
        query_idx = self.paper_ids_idx_mapping[query_paper["ids"]]
        pos_idx = self.paper_ids_idx_mapping[pos_paper["ids"]]
        neg_idx = self.paper_ids_idx_mapping[neg_paper["ids"]]

        query_pos_cos_sim = cosine_similarity(
            self.doc_embedding[query_idx], self.doc_embedding[pos_idx]
        )
        query_neg_cos_sim = cosine_similarity(
            self.doc_embedding[query_idx], self.doc_embedding[neg_idx]
        )

        return {
            "query_pos_abstract_jaccard": query_pos_abstract_jaccard,
            "query_neg_abstract_jaccard": query_neg_abstract_jaccard,
            "query_pos_abstract_inter_feature": query_pos_abstract_inter_feature,
            "query_neg_abstract_inter_feature": query_neg_abstract_inter_feature,
            "query_pos_cos_sim": query_pos_cos_sim,
            "query_neg_cos_sim": query_neg_cos_sim,
        }

    @staticmethod
    def _jaccard(tokenized_text_1: List[str], tokenized_text_2: List[str]):
        tokenized_text_1_set = set(tokenized_text_1)
        tokenized_text_2_set = set(tokenized_text_2)

        union = tokenized_text_1_set.union(tokenized_text_2_set)
        intersection = tokenized_text_1_set.intersection(tokenized_text_2_set)

        if len(union) > 0:
            return len(intersection) / len(union)
        else:
            return 0

    def _intersection_feature(
        self, tokenized_text_1: List[str], tokenized_text_2: List[str]
    ):
        tokenized_text_1_set = set(tokenized_text_1)
        tokenized_text_2_set = set(tokenized_text_2)

        intersection = tokenized_text_1_set.intersection(tokenized_text_2_set)

        intersection_feature = np.empty((len(intersection), self.model.dim))
        for i, word in enumerate(intersection):
            intersection_feature[i, :] = self.model.get_word_vector(word)

        intersection_feature = np.linalg.norm(
            np.sum(intersection_feature, axis=0), axis=1
        )

        return intersection_feature
