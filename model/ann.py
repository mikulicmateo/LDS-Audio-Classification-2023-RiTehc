import faiss
import numpy as np
import pandas as pd

DATA_PATH = '/home/dominik/Work/Lumen Datascience/LDS-Audio-Classification-2023-RiTehc/embeddings.pickle'

data = pd.read_pickle(DATA_PATH)

features = data[['embedding']].to_numpy()
x = np.array([features[i][0] for i in range(len(features))])

n = x.shape[0]
dimension = x.shape[1]
nlist = 11  # number of clusters

quantiser = faiss.IndexFlatL2(dimension)
index = faiss.IndexIVFFlat(quantiser, dimension, nlist, faiss.METRIC_L2)

index.train(x)
index.add(x)

# return k nearest neighbours
k = 5
query_vectors = x[15:30]
distances, indices = index.search(query_vectors, k)

