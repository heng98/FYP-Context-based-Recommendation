import torch


class Ranker:
    """Currently for simple reranker"""

    def __init__(self, reranker_model, doc_embedding_vectors, device):
        self.reranker_model = reranker_model
        self.doc_embedding_vectors = doc_embedding_vectors
        self.device = device

    def rank(self, query_embedding, candidates):
        candidates_ids = [c[0] for c in candidates]
        candidates_vector = torch.from_numpy(
            self.doc_embedding_vectors[candidates_ids]
        ).to(self.device)

        similarity = (
            self.reranker_model(
                query_embedding.expand(len(candidates_ids), -1), candidates_vector
            )
            .tolist()
        )

        reranked_candidates = [
            (ids, sim) for ids, sim in zip(candidates_ids, similarity)
        ]

        return sorted(reranked_candidates, key=lambda x: x[1], reverse=True)