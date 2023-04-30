import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

def get_scores(y_true, y_pred, labels):
    print(classification_report(y_true, y_pred, target_names=labels))

def fc_training_embedding_load(data_path):
    data = pd.read_pickle(data_path)

    features = data[['embedding']].to_numpy()
    labels = data[['label']].to_numpy()
    features = np.array([features[i][0] for i in range(len(features))])
    # features = np.squeeze(features, axis=1)
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
            x.append(np.array(features[i][j]))

    return np.array(x, dtype=object), labels

def function(x):
    return -x + 1

def threshold(x, t):
    if x > t:
        return 1
    return 0

def calculate_weights(dists):
    dists[dists == 0.0] = 0.00001  # division by 0 handle for floats
    dists[dists == 0] = 1  # division by 0 handle for integers
    weights = 1.0 / dists
    return weights / np.sum(weights)

if __name__ == "__main__":
    DATA_PATH = '/home/mateo/Lumen-data-science/LDS-Audio-Classification-2023-RiTehc/data/embeddings-16x32-encoder.pickle'
    VAL_DATA_PATH = '/home/mateo/Lumen-data-science/LDS-Audio-Classification-2023-RiTehc/data/embeddings-windowed-16x32-encoder.pickle'

    features, labels = fc_training_embedding_load(DATA_PATH)

    val_features, val_labels = fc_validation_embedding_load(VAL_DATA_PATH)

    classes = ['tru', 'gac', 'sax', 'cel', 'flu', 'gel', 'vio', 'cla', 'pia', 'org', 'voi']
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
    print(features.shape)
    print(val_features.shape)
    # tree = AnnoyANN(features.shape[1])
    # tree.build(features, labels, number_of_trees=30, metric='euclidean')
    # tree.save_model('euclidean-encoder-16x32-t30.ann')
    k = 30

    tree = AnnoyANN(features.shape[1], labels, path='euclidean-encoder-16x32-t30.ann', metric='euclidean', prefault=True)
    counter = 0
    y_pred = []
    for i, val_feature in enumerate(val_features):
        prediction = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        window_results = []
        for window in val_feature:
            query_results, query_distances = tree.query(window, k=k)

            for j in range(len(query_results)):
                window_results.append([query_results[j], query_distances[j]])

        sorted_results = sorted(window_results, key=lambda t: t[1])

        sorted_results = np.array(sorted_results[:30])
        weights = calculate_weights(sorted_results[:, 1])
        weighted_results = weights * sorted_results[:, 0]
        prediction = np.sum(weighted_results)
        # print("_-__-_-____-_____")
        # print(prediction)
        # print(val_labels[i])
        # print(np.sum(weighted_results))
        # print(val_labels[i])
        # for results, distances in sorted_results:
        #     weighted_result = [x * function(sorted_results[j][1]) for x in sorted_results[j][0]]
        #     prediction = np.add(prediction, weighted_result)

        # prediction /= 10
        # print(prediction)
        prediction = [threshold(x, 0.35) for x in prediction]
        y_pred.append(prediction)
        #
        if np.array_equal(np.array(prediction), val_labels[i]):
            counter += 1
            # print(val_labels[i])
            # print(np.array(prediction))
            # print("---------------------")

    get_scores(val_labels, y_pred, classes)
    # print(counter / val_features.shape[0])