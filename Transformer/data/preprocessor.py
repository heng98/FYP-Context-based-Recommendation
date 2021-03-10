from typing import List

from torch.nn.functional import cosine_similarity
from torchtext.data.utils import get_tokenizer

import fasttext
import numpy as np


class Preprocessor:
    def __call__(self, query_paper, pos_paper, neg_paper):
        raise NotImplementedError


class DefaultPreprocessor(Preprocessor):
    def __call__(self, query_paper, pos_paper, neg_paper, margin):
        return {
            "query_paper": query_paper,
            "pos_paper": pos_paper,
            "neg_paper": neg_paper,
            "margin": margin
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
            self.doc_embedding[[query_idx]], self.doc_embedding[[pos_idx]]
        )
        query_neg_cos_sim = cosine_similarity(
            self.doc_embedding[[query_idx]], self.doc_embedding[[neg_idx]]
        )

        return {
            "query_title_embedding": self.model.get_sentence_vector(query_paper["title"]),
            "pos_title_embedding": self.model.get_sentence_vector(pos_paper["title"]),
            "neg_title_embedding": self.model.get_sentence_vector(neg_paper["title"]),
            "query_abstract_embedding": self.model.get_sentence_vector(query_paper["abstract"]),
            "pos_abstract_embedding": self.model.get_sentence_vector(pos_paper["abstract"]),
            "neg_abstract_embedding": self.model.get_sentence_vector(neg_paper["abstract"]),
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
            return np.array([len(intersection) / len(union)], dtype="float32")
        else:
            return np.array([0], dtype="float32")

    def _intersection_feature(
        self, tokenized_text_1: List[str], tokenized_text_2: List[str]
    ):
        tokenized_text_1_set = set(tokenized_text_1)
        tokenized_text_2_set = set(tokenized_text_2)

        intersection = tokenized_text_1_set.intersection(tokenized_text_2_set)
        if len(intersection) == 0:
            return np.zeros(self.model.get_dimension(), dtype="float32")

        intersection_feature = np.empty((len(intersection), self.model.get_dimension()), dtype="float32")
        for i, word in enumerate(intersection):
            intersection_feature[i, :] = self.model.get_word_vector(word)
        intersection_feature = np.sum(intersection_feature, axis=0) / (
            max(
                np.linalg.norm(
                    np.sum(intersection_feature, axis=0), 2
                ), 
                1e-8
            )
        )
            
        return intersection_feature
