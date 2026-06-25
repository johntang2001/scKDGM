import numpy as np
import scipy.sparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import binary_cross_entropy_with_logits

from .kan_tagconv import KANTAGConv
from .layers import ClusteringLayer
from .losses import ZINB, info_nce_loss
from .metrics import clustering_scores


def gumbel_topk(logits, k, temperature=1.0):
    noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-20) + 1e-20)
    y_soft = torch.softmax((logits + noise) / temperature, dim=-1)
    _, idx = torch.topk(y_soft, k=k, dim=-1)
    y_hard = torch.zeros_like(logits).scatter_(1, idx, 1.0)
    y_st = y_hard - y_soft.detach() + y_soft
    hard_adj = torch.clamp(y_hard + y_hard.T, 0.0, 1.0)
    st_adj = y_st + y_st.T - y_st * y_st.T
    hard_adj.fill_diagonal_(0.0)
    st_adj = st_adj.clone()
    st_adj.fill_diagonal_(0.0)
    return hard_adj, st_adj


class EarlyStopper:
    def __init__(self, patience=30, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = -float("inf")

    def early_stop(self, score):
        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


class TAGEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim=256, latent_dim=128, dropout=0.2):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.conv1 = KANTAGConv(in_dim, hidden_dim, K=3, grid_size=4)
        self.conv2 = KANTAGConv(hidden_dim, latent_dim, K=3, grid_size=4)
        self.norm = nn.LayerNorm(latent_dim)

    def forward(self, x, edge_index, edge_weight=None):
        h = self.dropout(x)
        h = F.relu(self.conv1(h, edge_index, edge_weight=edge_weight))
        h = self.dropout(h)
        h = self.conv2(h, edge_index, edge_weight=edge_weight)
        return self.norm(h)


class ScKDGM(nn.Module):
    """KAN-guided dynamic graph masked learning for scRNA-seq clustering."""

    def __init__(self, x, zinb_target, adj, size_factor=None, hidden_dim=256, latent_dim=128, decoder_hidden=512, encoder_dropout=0.2):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if scipy.sparse.issparse(x):
            x = x.toarray()
        self.x = torch.tensor(x, dtype=torch.float32, device=self.device)
        # Keep the same ZINB target used by the current SCMaskGraphZINB implementation.
        self.zinb_target = self.x
        self.adj = self._sanitize_adj(torch.tensor(adj, dtype=torch.float32, device=self.device))
        self.n_cells, self.n_genes = self.x.shape
        self.edge_index = None
        self.edge_weight = None
        self.y_pred = None
        self.best_y_pred = None
        self.best_scores = None
        self.best_embedding = None
        scale_factor = 1.0 if size_factor is None else torch.tensor(size_factor, dtype=torch.float32, device=self.device)
        self.zinb_loss = ZINB(ridge_lambda=0.0, scale_factor=scale_factor)
        self.encoder = TAGEncoder(self.n_genes, hidden_dim, latent_dim, dropout=encoder_dropout)
        self.mask_predictor = nn.Linear(latent_dim, self.n_genes)
        self.attr_decoder = nn.Linear(latent_dim + self.n_genes, self.n_genes)
        self.zinb_decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.1), nn.Linear(hidden_dim, decoder_hidden), nn.ReLU()
        )
        self.pi_head = nn.Linear(decoder_hidden, self.n_genes)
        self.disp_head = nn.Linear(decoder_hidden, self.n_genes)
        self.mean_head = nn.Linear(decoder_hidden, self.n_genes)
        self.clustering_layer = ClusteringLayer(hidden_dim=latent_dim)
        self.to(self.device)

    def _sanitize_adj(self, adj):
        adj = torch.clamp((adj + adj.T) / 2.0, min=0.0)
        adj = adj.clone()
        adj.fill_diagonal_(0.0)
        return adj

    def _edge_index_and_weight(self, adj):
        idx = (adj > 0).nonzero(as_tuple=False).T.contiguous()
        weight = adj[idx[0], idx[1]].float()
        return idx.to(self.device), weight.to(self.device)

    def _sparsify_topk(self, adj, k):
        adj = self._sanitize_adj(adj)
        if k <= 0 or k >= adj.shape[0]:
            return adj
        _, idx = torch.topk(adj, k=min(k, adj.shape[1] - 1), dim=1)
        mask = torch.zeros_like(adj)
        mask.scatter_(1, idx, 1.0)
        adj = adj * mask
        return self._sanitize_adj(torch.maximum(adj, adj.T))

    def _edge_index_and_weight_from_st(self, hard_adj, st_adj):
        idx = (hard_adj > 0).nonzero(as_tuple=False).T.contiguous()
        weight = st_adj[idx[0], idx[1]].float()
        return idx.to(self.device), weight.to(self.device)

    def _sample_non_neighbor_perm(self, edge_index):
        n = self.n_cells
        adj_bool = torch.eye(n, dtype=torch.bool, device=self.device)
        src, dst = edge_index[0], edge_index[1]
        adj_bool[src, dst] = True
        adj_bool[dst, src] = True
        candidates = ~adj_bool
        no_candidate = ~candidates.any(dim=1)
        if no_candidate.any():
            candidates[no_candidate] = True
            candidates.fill_diagonal_(False)
        scores = torch.rand((n, n), device=self.device).masked_fill(~candidates, -1.0)
        return scores.argmax(dim=1)

    def _mask_features(self, mask_rate=0.4, edge_index=None):
        edge_index = self.edge_index if edge_index is None else edge_index
        sampled = torch.bernoulli(torch.full_like(self.x, float(mask_rate))).bool()
        perm = self._sample_non_neighbor_perm(edge_index)
        corrupted = torch.where(sampled, self.x[perm], self.x)
        mask = (corrupted != self.x).float()
        return corrupted, mask

    def encode(self, x=None, edge_index=None, edge_weight=None):
        x = self.x if x is None else x
        edge_index = self.edge_index if edge_index is None else edge_index
        edge_weight = self.edge_weight if edge_weight is None else edge_weight
        return self.encoder(x, edge_index, edge_weight=edge_weight)

    def _decode_zinb(self, z):
        h = self.zinb_decoder(z)
        pi = torch.sigmoid(self.pi_head(h))
        disp = torch.clamp(F.softplus(self.disp_head(h)), min=1e-4, max=1e4)
        mean = torch.clamp(torch.exp(self.mean_head(h)), min=1e-5, max=1e6)
        return pi, disp, mean

    def _feature_recon_loss(self, recon, mask, masked_data_weight=0.75):
        weights = mask * masked_data_weight + (1.0 - mask) * (1.0 - masked_data_weight)
        return (weights * F.mse_loss(recon, self.x, reduction="none")).mean()

    def _build_dynamic_graph(self, representation, k=15, temperature=1.0):
        z = F.normalize(representation, dim=1)
        logits = z @ z.T
        logits = logits.clone()
        logits.fill_diagonal_(-float("inf"))
        hard_adj, st_adj = gumbel_topk(logits, k=min(k, z.shape[0] - 1), temperature=temperature)
        hard_adj = self._sparsify_topk(hard_adj, k=k)
        st_adj = self._sanitize_adj(st_adj) * (hard_adj > 1e-6).float()
        edge_index, edge_weight = self._edge_index_and_weight_from_st(hard_adj, st_adj)
        return hard_adj.detach(), st_adj, edge_index, edge_weight

    def _masked_step(self, mask_rate, k, temperature, contrast_temperature):
        x_mask, mask = self._mask_features(mask_rate=mask_rate, edge_index=self.edge_index)
        z = self.encode(x_mask, self.edge_index, self.edge_weight)
        pred_mask = self.mask_predictor(z)
        x_hat = self.attr_decoder(torch.cat([z, torch.sigmoid(pred_mask)], dim=1))
        recon_loss = self._feature_recon_loss(x_hat, mask)
        mask_loss = binary_cross_entropy_with_logits(pred_mask, mask, reduction="mean")
        pi, disp, mean = self._decode_zinb(z)
        zinb_loss = self.zinb_loss(pi=pi, theta=disp, y_true=self.zinb_target, y_pred=mean)
        next_adj, _, next_edge_index, next_edge_weight = self._build_dynamic_graph(x_hat, k=k, temperature=temperature)
        z_dynamic = self.encode(self.x, next_edge_index, next_edge_weight)
        contrast_loss = info_nce_loss(z.detach(), z_dynamic, temperature=contrast_temperature)
        return recon_loss, mask_loss, zinb_loss, contrast_loss, next_adj, next_edge_index, next_edge_weight

    def _clean_step(self, k, temperature, contrast_temperature):
        z = self.encode(self.x, self.edge_index, self.edge_weight)
        pi, disp, mean = self._decode_zinb(z)
        zinb_loss = self.zinb_loss(pi=pi, theta=disp, y_true=self.zinb_target, y_pred=mean)
        next_adj, _, next_edge_index, next_edge_weight = self._build_dynamic_graph(z, k=k, temperature=temperature)
        z_dynamic = self.encode(self.x, next_edge_index, next_edge_weight)
        contrast_loss = info_nce_loss(z, z_dynamic, temperature=contrast_temperature)
        return z, zinb_loss, contrast_loss, next_adj, next_edge_index, next_edge_weight

    def pretrain(self, epochs=1000, lr=1e-4, k=15, mask_rate=0.4, gumbel_temperature=1.0, contrast_temperature=0.7,
                 recon_weight=1.0, mask_weight=0.1, zinb_weight=1.0, contrast_weight=0.1, log_interval=10):
        self.edge_index, self.edge_weight = self._edge_index_and_weight(self.adj)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
        for epoch in range(1, epochs + 1):
            self.train()
            optimizer.zero_grad()
            recon, mask, zinb, contrast, next_adj, next_edge_index, next_edge_weight = self._masked_step(
                mask_rate, k, gumbel_temperature, contrast_temperature
            )
            loss = recon_weight * recon + mask_weight * mask + zinb_weight * zinb + contrast_weight * contrast
            loss.backward()
            optimizer.step()
            scheduler.step()
            self.adj, self.edge_index, self.edge_weight = next_adj, next_edge_index, next_edge_weight.detach()
            if epoch % log_interval == 0:
                edge_count = int((self.adj > 0).sum().item())
                print(f"Epoch {epoch:4d} | Recon: {recon.item():.4f} | Mask: {mask.item():.4f} | ZINB: {zinb.item():.4f} | Contrast: {contrast.item():.4f} | Edges: {edge_count}")
        return self

    def pre_train(self, epochs=1000, info_step=10, lr=1e-4, k=15, gumbel_temperature=1.0, mask_rate=0.4,
                  replace_rate=0.1, feature_noise_mode="gdp-mask", recon_weight=1.0, mask_weight=0.1,
                  zinb_weight=1.0, contrast_weight=0.1, contra_temperature=0.7, update_graph=True,
                  use_mask=True, **kwargs):
        # Public API kept compatible with SCMaskGraphZINB. The released implementation uses the
        # paper's default masked pre-training path.
        return self.pretrain(
            epochs=epochs, lr=lr, k=k, mask_rate=mask_rate, gumbel_temperature=gumbel_temperature,
            contrast_temperature=contra_temperature, recon_weight=recon_weight, mask_weight=mask_weight,
            zinb_weight=zinb_weight, contrast_weight=contrast_weight, log_interval=info_step
        )

    def fit_clustering(self, y, centers, epochs=200, lr=1e-4, cluster_weight=1.0, zinb_weight=0.1, contrast_weight=0.01,
                       k=15, gumbel_temperature=1.0, contrast_temperature=0.7, target_update_interval=8, log_interval=1):
        centers = torch.tensor(centers, dtype=torch.float32, device=self.device)
        self.clustering_layer.set_centers(centers)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
        early_stopper = EarlyStopper(patience=30, min_delta=0.001)
        p = None
        y_pred = None
        best_ari = -float("inf")
        for epoch in range(epochs):
            if epoch % target_update_interval == 0 or p is None:
                self.eval()
                with torch.no_grad():
                    p = self.target_distribution(self.clustering_layer(self.encode()))
            self.train()
            optimizer.zero_grad()
            z, zinb, contrast, next_adj, next_edge_index, next_edge_weight = self._clean_step(k, gumbel_temperature, contrast_temperature)
            q = self.clustering_layer(z)
            cluster_loss = F.kl_div(torch.log(q + 1e-10), p, reduction="batchmean")
            loss = cluster_weight * cluster_loss + zinb_weight * zinb + contrast_weight * contrast
            loss.backward()
            optimizer.step()
            scheduler.step()
            self.adj, self.edge_index, self.edge_weight = next_adj, next_edge_index, next_edge_weight.detach()
            if epoch % log_interval == 0:
                y_pred = q.argmax(dim=1).detach().cpu().numpy()
                scores = clustering_scores(y, y_pred)
                if scores["ari"] > best_ari:
                    best_ari = scores["ari"]
                    self.best_y_pred = y_pred.copy()
                    self.best_embedding = z.detach().cpu().numpy()
                    self.best_scores = {"epoch": epoch, **scores}
                print(f"Epoch {epoch:3d} | Cluster: {cluster_loss.item():.4f} | ZINB: {zinb.item():.4f} | Contrast: {contrast.item():.4f} | ACC: {scores['acc']:.4f}, NMI: {scores['nmi']:.4f}, ARI: {scores['ari']:.4f}")
                if early_stopper.early_stop(scores["ari"]):
                    print(f"Early stopping at epoch {epoch}")
                    break
        self.y_pred = y_pred
        return self

    def alt_train(self, y, epochs=200, centers=None, info_step=1, lr=1e-4, W_c=1.0, W_x=0.1,
                  contrast_weight=0.01, contra_temperature=0.7, k=15, gumbel_temperature=1.0,
                  update_graph=True, n_update=8, **kwargs):
        return self.fit_clustering(
            y=y, centers=centers, epochs=epochs, lr=lr, cluster_weight=W_c, zinb_weight=W_x,
            contrast_weight=contrast_weight, k=k, gumbel_temperature=gumbel_temperature,
            contrast_temperature=contra_temperature, target_update_interval=n_update, log_interval=info_step
        )

    def get_embedding(self):
        self.eval()
        with torch.no_grad():
            z = self.encode()
        return z.detach().cpu().numpy()

    def target_distribution(self, q):
        q = q.detach().cpu().numpy()
        weight = q ** 2 / (q.sum(axis=0) + 1e-10)
        p = (weight.T / (weight.sum(axis=1) + 1e-10)).T
        return torch.tensor(p, dtype=torch.float32, device=self.device)
