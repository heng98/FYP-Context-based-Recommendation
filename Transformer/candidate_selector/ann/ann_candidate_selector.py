import numpy as np


class ANNCandidateSelector:
    def __init__(
        self, ann, neighbour_candidate, train_paper_dataset, ids_idx, extend_candidate=True
    ):
        self.ann = ann
        self.neighbour_candidate = neighbour_candidate
        self.extend_candidate = extend_candidate

        self.train_paper_dataset = train_paper_dataset
        self.candidate = set(train_paper_dataset)
        self.ids_idx = ids_idx
        self.idx_ids = list(ids_idx.keys())

    def get_candidate(self, query_embedding):
        candidate = self.ann.get_k_nearest_neighbour(
            query_embedding, self.neighbour_candidate
        )
        if self.extend_candidate:
            candidate_set = set(candidate)
            for i in candidate:
                citation_of_nn = self.train_paper_dataset[self.idx_ids[i]]["pos"]
                candidate_set.update([self.ids_idx[c] for c in citation_of_nn if c in self.candidate])

            candidate = list(candidate_set)

        similarity = self._get_similarity(query_embedding, candidate)

        result = [(self.idx_ids[idx], sim) for idx, sim in zip(candidate, similarity)]
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
