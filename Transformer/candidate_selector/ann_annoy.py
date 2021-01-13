from annoy import AnnoyIndex


class ANNAnnoy:
    def __init__(self, index):
        self.index = index

    @classmethod
    def build_graph(cls, doc_embedding_vectors, ann_trees=100):
        embedding_dim = doc_embedding_vectors.shape[1]
        index = AnnoyIndex(embedding_dim, 'angular')

        for i, doc_embedding in enumerate(doc_embedding_vectors):
            index.add_item(i, doc_embedding)

        index.build(ann_trees)

        return cls(index)

    def load_graph(self):
        pass

    def get_k_nearest_neighbour(self, query_vector, top_k):
        top_k_result = self.index.get_nns_by_vector(
            query_vector, top_k, include_distances=True
        )

        return top_k_result
