import torch
from torchtext.data.utils import get_tokenizer

import numpy as np
import fasttext


class Ranker:
    """Currently for simple reranker"""

    def __init__(self, reranker_model, doc_embedding_vectors, device, fasttext_path):
        self.reranker_model = reranker_model
        self.doc_embedding_vectors = doc_embedding_vectors
        self.device = device

        self.model = fasttext.load_model(fasttext_path)
        self.tokenizer = get_tokenizer("basic_english")

    def rank(self, query_embedding, candidates, query_data, candidate_data):
        candidates_idx = [c[0] for c in candidates]
        query_embedding = query_embedding.expand(len(candidates), -1).to(self.device)
        candidates_embedding = torch.from_numpy(
            self.doc_embedding_vectors[candidates_idx]
        ).to(self.device)

        cos_similarity = (
            torch.nn.functional.cosine_similarity(query_embedding, candidates_embedding)
            .unsqueeze(0)
            .T
        )

        tokenized_query_abstract = self.tokenizer(query_data["abstract"])
        tokenized_candidate_abstract = [
            self.tokenizer(c["abstract"]) for c in candidate_data
        ]

        jaccard = self._jaccard(tokenized_query_abstract, tokenized_candidate_abstract)
        jaccard = torch.tensor(jaccard, device=self.device).unsqueeze(0).T

        intersection_feature = torch.from_numpy(
            self._intersection_feature(
                tokenized_query_abstract, tokenized_candidate_abstract
            )
        )

        confidence = self.reranker_model(jaccard, intersection_feature, cos_similarity)
        confidence = confidence.flatten().tolist()

        reranked_candidates = [
            (c["ids"], conf) for c, conf in zip(candidate_data, confidence)
        ]

        return sorted(reranked_candidates, key=lambda x: x[1], reverse=True)

    def _jaccard(self, text_1, text_2):
        result = []
        tokenized_t1 = set(text_1)
        tokenized_t2 = [set(t) for t in text_2]

        for t2 in tokenized_t2:
            union = len(set(tokenized_t1).union(set(t2)))
            if union > 0:
                result.append(len(set(tokenized_t1).intersection(set(t2))) / union)
            else:
                result.append(0)

        return result

    def _intersection_feature(self, text_1, text_2):
        tokenized_text_1_set = set(text_1)
        tokenized_text_2_set = [set(t2) for t2 in text_2]

        result_intersection_feature = np.empty(len(text_2), self.model.dim)

        for j, t2 in enumerate(tokenized_text_2_set):
            intersection = tokenized_text_1_set.intersection(t2)

            intersection_feature = np.empty((len(intersection), self.model.dim))
            for i, word in enumerate(intersection):
                intersection_feature[i, :] = self.model.get_word_vector(word)

            intersection_feature = np.linalg.norm(
                np.sum(intersection_feature, axis=0), axis=1
            )

            result_intersection_feature[j, :] = intersection_feature

        return result_intersection_feature


class TransformerRanker:
    def __init__(self, reranker_model, device, tokenizer):
        self.reranker_model = reranker_model
        self.device = device
        self.tokenizer = tokenizer

    def rank(self, query, candidates):
        query_text = query["title"] + query["abstract"]
        candidates_text = [c["title"] + c["abstract"] for c in candidates]
        candidates_ids = [c["ids"] for c in candidates]

        query_encoded = self._encode(query_text)
        candidates_encoded = self._encode(candidates_text)
        candidates_encoded["input_ids"][:, 0] = self.tokenizer.sep_token_id

        for k in query_encoded:
            query_encoded[k] = query_encoded[k].expand_as(candidates_encoded[k])

        combined_encoded = {
            k: torch.cat([query_encoded[k], candidates_encoded[k]], axis=1).to(
                self.device
            )
            for k in query_encoded
        }

        similarity = torch.flatten(
            torch.sigmoid(self.reranker_model(**combined_encoded)["logits"])
        )
        sorted_sim, indices = torch.sort(similarity, descending=True)

        sorted_sim = sorted_sim.tolist()
        indices = indices.tolist()

        reranked_candidates = [
            (candidates_ids[idx], sim) for idx, sim in zip(indices, sorted_sim)
        ]

        return reranked_candidates

    def _encode(self, text):
        return self.tokenizer(
            text,
            padding="max_length",
            max_length=256,
            truncation=True,
            return_tensors="pt",
        )
