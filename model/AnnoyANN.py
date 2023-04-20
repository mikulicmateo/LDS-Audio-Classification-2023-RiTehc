from annoy import AnnoyIndex


class AnnoyANN:
    def __init__(self, dimension, labels=None, metric='angular', path=None, prefault=False):
        self.dimension = dimension
        self.labels = labels
        if path:
            self.index = AnnoyIndex(dimension, metric=metric)
            self.index.load(path, prefault=prefault)


    def build(self, vectors, labels, number_of_trees=11, metric='angular'):
        self.vectors = vectors.astype('float32')
        self.labels = labels
        self.index = AnnoyIndex(self.dimension, metric=metric)

        for i, vec in enumerate(self.vectors):
            self.index.add_item(i, vec.tolist())

        self.index.build(number_of_trees)

    def query(self, vector, k=10, search_in_x_trees=None):
        if search_in_x_trees:
            indices, distances = self.index.get_nns_by_vector(
                vector.tolist(),
                k,
                search_k=search_in_x_trees,
                include_distances=True)
        else:
            indices, distances = self.index.get_nns_by_vector(vector.tolist(), k, include_distances=True)

        return [self.labels[i] for i in indices], distances

    def query_without_distances(self, vector, k=10, search_in_x_trees=None):
        if search_in_x_trees:
            indices, distances = self.index.get_nns_by_vector(
                vector.tolist(),
                k,
                search_k=search_in_x_trees)
        else:
            indices, distances = self.index.get_nns_by_vector(vector.tolist(), k)

        return [self.labels[i] for i in indices]

    def save_model(self, path):
        self.index.save(path)
