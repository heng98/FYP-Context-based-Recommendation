from annoy import AnnoyIndex


class ANNAnnoy:
    def __init__(self, index):
        self.index = index

    @classmethod
    def build(cls, doc_embedding_vectors, ann_trees=100):
        embedding_dim = doc_embedding_vectors.shape[1]
        index = AnnoyIndex(embedding_dim, "euclidean")

        for i, doc_embedding in enumerate(doc_embedding_vectors):
            index.add_item(i, doc_embedding)

        index.build(ann_trees)

        return cls(index)

    def save(self, path):
        self.index.save(path)

    @classmethod
    def load(cls, path, embedding_dim=768):
        index = AnnoyIndex(embedding_dim, "euclidean")
        index.load(path)

        return cls(index)

    def get_k_nearest_neighbour(self, query_vector, top_k):
        top_k_result = self.index.get_nns_by_vector(
            query_vector, top_k, include_distances=False
        )

        return top_k_result

    def get_items_vector(self, items_list):
        vectors = []
        for item in items_list:
            vectors.append(self.index.get_item_vector(item))

        return vectors
