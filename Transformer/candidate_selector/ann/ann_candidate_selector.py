import numpy as np


class ANNCandidateSelector:
    def __init__(
        self,
        ann,
        num_neighbour_candidate,
        corpus,
        paper_ids_idx_mapping,
        candidate_pool,
        extend_candidate=True,
    ):
        self.ann = ann
        self.num_neighbour_candidate = num_neighbour_candidate
        self.extend_candidate = extend_candidate

        self.corpus = corpus
        self.paper_ids_idx_mapping = paper_ids_idx_mapping
        self.candidate_pool = candidate_pool

        self.idx_paper_ids_mapping = list(self.paper_ids_idx_mapping.keys())

    def get_candidate(self, query_embedding):
        candidate = self.ann.get_k_nearest_neighbour(
            query_embedding, self.num_neighbour_candidate
        )
        if self.extend_candidate:
            candidate_set = set(candidate)
            for i in candidate:
                citation_of_nn = self.corpus[self.idx_paper_ids_mapping[i]]["pos"]
                candidate_set.update(
                    [
                        self.paper_ids_idx_mapping[c]
                        for c in citation_of_nn
                        if c in self.candidate_pool
                    ]
                )

            candidate = list(candidate_set)

        similarity = self._get_similarity(query_embedding, candidate)

        result = [
            (self.idx_paper_ids_mapping[idx], sim)
            for idx, sim in zip(candidate, similarity)
        ]
        result = sorted(result, key=lambda x: x[1])

        return result

    def _get_similarity(self, query_embedding, candidate_list):
        candidate_vector = np.array(self.ann.get_items_vector(candidate_list))

        # sim = np.dot(candidate_vector, query_embedding) / (
        #     np.linalg.norm(candidate_vector, 2, axis=1)
        #     * np.linalg.norm(query_embedding, 2)
        # )
        sim = np.linalg.norm(candidate_vector - query_embedding, 2, axis=1)

        return sim.tolist()
