import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, hamming_loss, accuracy_score
import matplotlib.pyplot as plt

def get_scores(y_true, y_pred, labels):
    print(classification_report(y_true, y_pred, target_names=labels))
    print("HAMMING_SCORE ------", 1 - hamming_loss(y_true, y_pred))
    print("ACCURACY_SCORE -----",  accuracy_score(y_true, y_pred))

def fc_training_embedding_load(data_path):
    data = pd.read_pickle(data_path)

    #features = data[['embedding']].to_numpy()
    labels = data[['label']].to_numpy()
    #features = np.array([features[i][0] for i in range(len(features))])
    # features = np.squeeze(features, axis=1)
    labels = np.array([labels[i][0] for i in range(len(labels))])
    print(labels.shape)
    return labels#features, labels

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
    if x >= t:
        return 1
    return 0

def calculate_weights(dists):
    dists[dists == 0.0] = 0.00001  # division by 0 handle for floats
    dists[dists == 0] = 1  # division by 0 handle for integers
    weights = 1.0 / dists
    return weights / np.sum(weights)

if __name__ == "__main__":
    DATA_PATH = '/home/mateo/Lumen-data-science/LDS-Audio-Classification-2023-RiTehc/app/api/data/labels-256-resnet34.pickle'
    VAL_DATA_PATH = '/home/mateo/Lumen-data-science/LDS-Audio-Classification-2023-RiTehc/data/embeddings-windowed-512-resnet34.pickle'

    #features, \
    labels = fc_training_embedding_load(DATA_PATH)

    val_features, val_labels = fc_validation_embedding_load(VAL_DATA_PATH)
    # val_features = val_features[:800]
    # val_labels = val_labels[:800]
    classes = ['tru', 'gac', 'sax', 'cel', 'flu', 'gel', 'vio', 'cla', 'pia', 'org', 'voi']
    """
        FAISS
    """
    import faiss

    n = val_features.shape[0]
    #dimension = val_features.shape[1]
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
    # print(features.shape)
    print(val_features.shape)
    # tree = AnnoyANN(features.shape[1])
    # tree.build(features, labels, number_of_trees=30, metric='angular')
    # tree.save_model('angular-resnet34-uniform-256-t30.ann')
    k = 30

    tree = AnnoyANN(256, labels, path='angular-resnet34-256-t30.ann', metric='angular', prefault=True)

    for p in range(1, 10):
        y_pred = []
        for val_feature in val_features:
            sample_prediction = [0 for _ in classes]
            window_results = []
            for j, window in enumerate(val_feature):
                window_prediction = [0 for _ in classes]
                query_results, query_distances = tree.query(window, k=k)

                query_results = np.array(query_results)
                weights = calculate_weights(np.array(query_distances))
                weighted_results = [1. * query_result for weight, query_result in zip(weights, query_results)]

                for result in weighted_results:
                    window_prediction = np.add(window_prediction, result)

                window_prediction /= k
                window_results = [threshold(x, float(p) / 100. + 0.9) for x in window_prediction]
                sample_prediction = [x + y for x, y in zip(sample_prediction, window_results)]

            for i, x in enumerate(sample_prediction):
                if x > 1:
                    sample_prediction[i] = 1

            y_pred.append(sample_prediction)

            # plt.title(f"{i + 1}. File")
            # plt.bar(classes, sample_prediction, width=0.9)
            # plt.show()
        #
        print(f"{float(p) / 100. + 0.9}:")
        get_scores(val_labels, np.array(y_pred), classes)
        #angular
        #Best HAMMING_SCORE ------ 0.07625, k=30 p = 0.38, no weights
        #Best HAMMING_SCORE ------ 0.07579545454545454, k=10, p=0.41, no weights
        #Best HAMMING_SCORE ------ 0.07625, k=10, p=0.043, weights
        #Best HAMMING_SCORE ------ 0.07602272727272727, k=30, p=0.013, weights
        #euclidean
        #Best HAMMING_SCORE ------ 0.075, k=10, p=0.5, no weights