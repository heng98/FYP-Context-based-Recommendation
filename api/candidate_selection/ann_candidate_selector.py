import numpy as np


class ANNCandidateSelector:
    def __init__(self, ann, neighbour_candidate, corpus, extend_candidate=True):
        self.ann = ann
        self.neighbour_candidate = neighbour_candidate
        self.extend_candidate = extend_candidate

        self.corpus = corpus

    def get_candidate(self, query_embedding):
        candidate_ann_idx = self.ann.get_k_nearest_neighbour(
            query_embedding, self.neighbour_candidate
        )

        candidate_paper_ids = self.corpus.get_paper_ids_by_ann_ids_list(
            candidate_ann_idx
        )

        if self.extend_candidate:
            citation_of_candidate = self.corpus.get_citation_by_ids_list(
                candidate_paper_ids
            )

        candidate_paper_ids += citation_of_candidate

        candidate_papers = self.corpus.get_paper_by_ids_list(candidate_paper_ids)
        candidate_ann_idx = [paper["ann_id"] for paper in candidate_papers]
        
        similarity = self._get_distance(query_embedding, candidate_ann_idx)

        result = [
            (paper, sim)
            for paper, sim in zip(candidate_papers, similarity)
        ]
        result = sorted(result, key=lambda x: x[1])

        return result

    def _get_distance(self, query_embedding, candidate_list):
        candidate_vector = np.array(self.ann.get_items_vector(candidate_list))
        dist = np.linalg.norm(candidate_vector - query_embedding, 2, axis=1)

        return dist.tolist()