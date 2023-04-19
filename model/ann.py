import faiss
import numpy as np
import pandas as pd

DATA_PATH = '/home/dominik/Work/Lumen Datascience/LDS-Audio-Classification-2023-RiTehc/embeddings.pickle'

data = pd.read_pickle(DATA_PATH)

features = data[['embedding']].to_numpy()
features = np.array([features[i][0] for i in range(len(features))])

n = features.shape[0]
dimension = features.shape[1]
nlist = 11  # number of clusters

quantiser = faiss.IndexFlatL2(dimension)
index = faiss.IndexIVFFlat(quantiser, dimension, nlist, faiss.METRIC_L2)

index.train(features)
index.add(features)

# return k nearest neighbours
k = 5
query_vectors = features[15:30]
distances, indices = index.search(query_vectors, k)

labels = data[['label']].to_numpy()

y = []
for i in range(len(labels)):
    y.append(labels[i][0])

labels = np.array(y)
"""
        ANNOY
"""
from AnnoyANN import AnnoyANN

tree = AnnoyANN(features, labels)
tree.build()
query_results, query_distances = tree.query(features[0])

for i, distance in enumerate(query_distances):
    print(f'Similar label: {query_results[i]}, hamming distance: {distance}')


