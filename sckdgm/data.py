import h5py
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp


def _decode(values):
    values = np.asarray(values)
    if values.dtype.type is np.bytes_:
        return np.asarray([v.decode("utf-8") for v in values])
    return values


def _read_item(obj):
    if isinstance(obj, h5py.Group):
        return {key: _read_item(obj[key]) for key in obj.keys()}
    return _decode(obj[...])


def read_h5_dataset(path):
    with h5py.File(path, "r") as f:
        obs = pd.DataFrame(_read_item(f["obs"]), index=_decode(f["obs_names"][...]))
        exprs = f["exprs"]
        if isinstance(exprs, h5py.Group):
            x = sp.csr_matrix((exprs["data"][...], exprs["indices"][...], exprs["indptr"][...]), shape=exprs["shape"][...])
            x = x.toarray()
        else:
            x = exprs[...]
    labels = np.unique(np.asarray(obs["cell_type1"]), return_inverse=True)[1]
    return np.asarray(x, dtype=np.float32), labels.astype(np.int64)


def preprocess_counts(counts, n_hvg=1000):
    counts = np.ceil(np.asarray(counts, dtype=np.float32))
    adata = sc.AnnData(counts)
    sc.pp.filter_genes(adata, min_counts=1)
    sc.pp.filter_cells(adata, min_counts=1)
    adata.raw = adata.copy()
    sc.pp.normalize_per_cell(adata)
    adata.obs["size_factors"] = adata.obs.n_counts / np.median(adata.obs.n_counts)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(
        adata,
        min_mean=0.0125,
        max_mean=3,
        min_disp=0.5,
        n_top_genes=n_hvg,
        subset=True,
    )
    hvg_index = np.asarray(adata.var.highly_variable.index, dtype=np.int64)
    count_hvg = counts[:, hvg_index].astype(np.float32)
    zinb_target = np.asarray(adata.X, dtype=np.float32)
    size_factor = np.asarray(adata.obs["size_factors"]).reshape(-1, 1).astype(np.float32)
    sc.pp.scale(adata)
    x = np.asarray(adata.X, dtype=np.float32)
    return x, zinb_target, count_hvg, size_factor
