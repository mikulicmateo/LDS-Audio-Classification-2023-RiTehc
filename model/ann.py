import numpy as np
import pandas as pd

def fc_training_embedding_load(data_path):
    data = pd.read_pickle(data_path)

    features = data[['embedding']].to_numpy()
    labels = data[['label']].to_numpy()
    features = np.array([features[i][0] for i in range(len(features))])
    features = np.squeeze(features, axis=1)
    labels = np.array([labels[i][0] for i in range(len(labels))])

    return features, labels

def fc_validation_embedding_load(data_path):
    data = pd.read_pickle(data_path)

    features = data[['embeddings']].to_numpy()
    labels = data[['labels']].to_numpy()
    labels = np.array([labels[i][0] for i in range(len(labels))])

    x = []
    for i in range(len(features)):
        for j in range(len(features[i])):
            x.append(np.squeeze(np.array(features[i][j]), axis=1))

    return np.array(x, dtype=object), labels


if __name__ == "__main__":
    DATA_PATH = '/home/mateo/Lumen-data-science/LDS-Audio-Classification-2023-RiTehc/data/embeddings-1024-fc.pickle'
    VAL_DATA_PATH = '/home/mateo/Lumen-data-science/LDS-Audio-Classification-2023-RiTehc/data/embeddings-val-1024-fc.pickle'

    features, labels = fc_training_embedding_load(DATA_PATH)

    val_features, val_labels = fc_validation_embedding_load(VAL_DATA_PATH)

    """
        FAISS
    """
    import faiss

    n = features.shape[0]
    dimension = features.shape[1]
    nlist = 11  # number of clusters

    # quantiser = faiss.IndexFlatL2(dimension)
    # index = faiss.IndexIVFFlat(quantiser, dimension, nlist, faiss.METRIC_L2)
    #
    # index.train(features)
    # index.add(features)
    #
    # # return k nearest neighbours
    # k = 5
    # query_vectors = features[15:30]
    # distances, indices = index.search(query_vectors, k)
    """
            ANNOY
    """
    from AnnoyANN import AnnoyANN

    tree = AnnoyANN(features, labels)
    tree.build(number_of_trees=1024, metric='angular')

    for i, val_feature in enumerate(val_features):
        print(val_labels[i])
        for window in val_feature:
            query_results, query_distances = tree.query(window, k=3)
            for i, distance in enumerate(query_distances):
                print(f'{i+1}. similar label: {query_results[i]}, distance: {distance}\n\t------')
