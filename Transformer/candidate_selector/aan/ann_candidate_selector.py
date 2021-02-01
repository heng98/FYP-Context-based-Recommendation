import numpy as np


class ANNCandidateSelector:
    def __init__(
        self, ann, neighbour_candidate, train_paper_dataset, extend_candidate=True
    ):
        self.ann = ann
        self.neighbour_candidate = neighbour_candidate
        self.extend_candidate = extend_candidate

        self.train_paper_dataset = train_paper_dataset

    def get_candidate(self, query_embedding, top_k):
        candidate = self.ann.get_k_nearest_neighbour(
            query_embedding, self.neighbour_candidate
        )

        if self.extend_candidate:
            candidate_set = set(candidate)
            for i in candidate:
                _, citation_of_nn = self.train_paper_dataset[i]
                candidate_set.update(citation_of_nn)

            candidate = list(candidate_set)

        similarity = self._get_similarity(query_embedding, candidate)

        result = [(ids, sim) for ids, sim in zip(candidate, similarity)]
        result = sorted(result, key=lambda x: x[1], reverse=True)

        return result

    def _get_similarity(self, query_embedding, candidate_list):
        candidate_vector = np.array(self.ann.get_items_vector(candidate_list))

        sim = np.dot(candidate_vector, query_embedding) / (
            np.linalg.norm(candidate_vector, 2, axis=1)
            * np.linalg.norm(query_embedding, 2)
        )

        return sim.tolist()
