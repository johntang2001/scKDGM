import argparse
import os
import random

import numpy as np
import torch
from sklearn.cluster import SpectralClustering

from sckdgm import ScKDGM
from sckdgm.data import preprocess_counts, read_h5_dataset
from sckdgm.graph import build_knn_graph
from sckdgm.metrics import clustering_scores


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_centers(embedding, labels):
    return np.vstack([embedding[labels == i].mean(axis=0) for i in np.unique(labels)])


def main():
    parser = argparse.ArgumentParser(description="Train scKDGM on one scRNA-seq dataset.")
    default_data = os.path.join(os.path.dirname(__file__), "data", "Quake_Smart-seq2_Diaphragm", "data.h5")
    parser.add_argument("--data", default=default_data)
    parser.add_argument("--hvg", type=int, default=1000)
    parser.add_argument("--k", type=int, default=15)
    parser.add_argument("--seed", type=int, default=1111)
    parser.add_argument("--pretrain-epochs", type=int, default=1000)
    parser.add_argument("--cluster-epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--mask-rate", type=float, default=0.3)
    parser.add_argument("--gumbel-temperature", type=float, default=1.0)
    args = parser.parse_args()

    set_seed(args.seed)
    counts, y = read_h5_dataset(args.data)
    x, _, size_factor = preprocess_counts(counts, n_hvg=args.hvg)
    n_clusters = len(np.unique(y))
    adj, _ = build_knn_graph(x, k=args.k, pca_dim=50)

    print(f"Dataset: {os.path.basename(os.path.dirname(args.data))}")
    print(f"Cells: {x.shape[0]}, HVGs: {x.shape[1]}, clusters: {n_clusters}")
    model = ScKDGM(x=x, adj=adj, size_factor=size_factor)
    print(model)

    print("\n========== Pre-training ==========")
    model.pretrain(
        epochs=args.pretrain_epochs,
        lr=args.lr,
        k=args.k,
        mask_rate=args.mask_rate,
        gumbel_temperature=args.gumbel_temperature,
        log_interval=10,
    )

    print("\n========== Initialize Clusters ==========")
    embedding = model.get_embedding()
    init_labels = SpectralClustering(
        n_clusters=n_clusters,
        affinity="precomputed",
        assign_labels="discretize",
        random_state=0,
    ).fit_predict(adj)
    centers = compute_centers(embedding, init_labels)

    print("\n========== Clustering ==========")
    model.fit_clustering(
        y=y,
        centers=centers,
        epochs=args.cluster_epochs,
        lr=args.lr,
        k=args.k,
        gumbel_temperature=args.gumbel_temperature,
        log_interval=1,
    )
    scores = clustering_scores(y, model.y_pred)
    print(f"\nFinal | ACC: {scores['acc']:.4f}, NMI: {scores['nmi']:.4f}, ARI: {scores['ari']:.4f}")


if __name__ == "__main__":
    main()
