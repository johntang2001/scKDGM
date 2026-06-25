# scKDGM

Official implementation of **scKDGM: KAN-guided Dynamic Graph Masked Learning for Single-Cell RNA-seq Clustering**.

This repository contains only the core implementation used for model training and evaluation. Extra visualization scripts, sweep scripts, ablation switches, and experimental logs are intentionally omitted for anonymous review.

## Contents

```text
scKDGM/
├── data/Quake_Smart-seq2_Diaphragm/data.h5
├── sckdgm/
│   ├── data.py
│   ├── graph.py
│   ├── kan_tagconv.py
│   ├── layers.py
│   ├── losses.py
│   ├── metrics.py
│   └── model.py
├── train.py
└── requirements.txt
```

## Installation

```bash
pip install -r requirements.txt
```

PyTorch and PyG installation can depend on CUDA version. If `pip install torch-geometric` is not sufficient on your machine, please follow the official PyG installation command matching your PyTorch/CUDA build.

## Quick Start

Run the included test dataset:

```bash
python train.py
```

Useful runtime options:

```bash
python train.py
```
