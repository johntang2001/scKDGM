import numpy as np
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.neighbors import kneighbors_graph


def normalize_adj(adj):
    degree = np.asarray(adj.sum(1)).reshape(-1)
    degree = np.power(degree, -0.5)
    degree[np.isinf(degree)] = 0.0
    d_mat = sparse.diags(degree) if sparse.issparse(adj) else np.diag(degree)
    return d_mat.dot(adj).dot(d_mat)


def build_knn_graph(x, k=15, pca_dim=50):
    x_graph = PCA(n_components=pca_dim).fit_transform(x) if pca_dim else x
    adj = kneighbors_graph(x_graph, k, mode="connectivity", metric="euclidean", include_self=True).toarray()
    adj = adj + adj.T
    adj[adj > 1] = 1
    return adj.astype(np.float32), normalize_adj(adj).astype(np.float32)
