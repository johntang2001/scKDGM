import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def cluster_acc(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    dim = max(y_pred.max(), y_true.max()) + 1
    weight = np.zeros((dim, dim), dtype=np.int64)
    for pred, true in zip(y_pred, y_true):
        weight[pred, true] += 1
    row, col = linear_sum_assignment(weight.max() - weight)
    return weight[row, col].sum() / y_pred.size


def clustering_scores(y_true, y_pred):
    return {
        "acc": cluster_acc(y_true, y_pred),
        "nmi": normalized_mutual_info_score(y_true, y_pred),
        "ari": adjusted_rand_score(y_true, y_pred),
    }
