# 🚀 SparseEIEA: A Memory-Efficient Approach for Multi-Modal Entity Alignment via Top-k Sparsification

![Version](https://img.shields.io/badge/version-1.0.0-blue)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **SparseEIEA** is a memory-efficient approach for **multi-modal entity alignment (MMEA)** in large-scale knowledge graphs.  
> It replaces dense representation with a deterministic **top-k sparsification** strategy that retains only the most promising candidate matches for each entity, reducing memory complexity from **O(N²)** to **O(N·k)** while maintaining comparable alignment accuracy.
>Combined with **Sparse Sinkhorn assignment** and **FP16 half-precision computation**, SparseEIEA achieves substantial memory savings while preserving alignment performance.
---

### Key Features

* **Sparse Profit Matrix** — Retains only the top-k candidate matches per entity, reducing memory complexity from \(O(N^2)\) to \(O(N \cdot k)\)
* **Sparse Sinkhorn Assignment** — Performs efficient sparse assignment with scalable optimization
* **FP16 Precision** — Uses half-precision computation to further reduce memory consumption
* **High Accuracy Preservation** — Preserves more than **98% of dense alignment accuracy**
* **Memory Efficient** — Achieves up to **83.34% memory reduction** in experimental settings
* **Scalable** — Handles **1 million entities using only 10.8 GB memory**, compared with approximately **3.6 TB** for dense matrix-based methods

---

## Framework

<div align="center">
<img src="picture/framework.png" width="100%" alt="SparseEIEA Framework"/>
</div>

SparseEIEA consists of three main stages:

1. **Multi-Modal Feature Processing**
   Multi-modal entity representations are obtained from available structural, visual, attribute, and name information.

2. **Top-k Candidate Sparsification**
   Instead of storing all possible entity pairs, only the top-k highest-scoring candidate matches are retained for each entity.

3. **Sparse Assignment**
   A sparse Sinkhorn-based assignment procedure is applied to efficiently estimate entity correspondences using the sparsified candidate matrix.

The overall workflow significantly reduces memory consumption while maintaining competitive entity alignment performance.

---

## Environment

### Requirements

* Python 3.8+
* PyTorch 1.10+
* CUDA 11.0+ (for GPU support)

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## Dataset

The processed **MMEA-data** and **ECS-results** used in SparseEIEA can be downloaded from:

[Google Drive](https://drive.google.com/drive/folders/1wfErYdAV93yxPtPHqkGanbmb_Ztv-LRU?usp=drive_link)

The original datasets and related resources are available from:

* [MMKB](https://github.com/mniepert/mmkb)
* [MMEA / MCLEA](https://github.com/lzxlin/mclea?tab=readme-ov-file)

After downloading the datasets, organize the files as follows:

```text
SparseEIEA/
├── data/
│   ├── ECS_results/
│   │   ├── seed0.2/
│   │   │   ├── FB15K-DB15K/
│   │   │   │   ├── Vis.npy
│   │   │   │   ├── Attr1.npy
│   │   │   │   └── name.npy
│   │   │   └── FB15K-YAGO15K/
│   │   │       ├── Vis.npy
│   │   │       ├── Attr1.npy
│   │   │       └── name.npy
│   │   ├── seed0.5/
│   │   │   ├── FB15K-DB15K/
│   │   │   └── FB15K-YAGO15K/
│   │   └── seed0.8/
│   │       ├── FB15K-DB15K/
│   │       └── FB15K-YAGO15K/
│   │
│   └── MMEA-data/
│       ├── seed0.2/
│       │   ├── FB15K-DB15K/
│       │   │   ├── triples_1
│       │   │   ├── triples_2
│       │   │   ├── sup_ent_ids
│       │   │   └── dev_ent_ids
│       │   └── FB15K-YAGO15K/
│       │
│       ├── seed0.5/
│       └── seed0.8/
│
├── src/
│   ├── sparse_eiea.py
│   ├── dataset.py
│   ├── util.py
│   └── memory_profiler.py
│
├── experiments/
│   └── run_benchmark.py
│
├── results/
│
└── requirements.txt
```

---

# Running SparseEIEA

## Basic Usage

Run SparseEIEA using:

```bash
python src/sparse_eiea.py --dataset FB15K-DB15K --seed 0.5
```

---

## Full Command with Options

```bash
python src/sparse_eiea.py \
    --dataset FB15K-DB15K \
    --seed 0.5 \
    --k_values 50 100 200 500 1000 \
    --device cuda:0
```

### Main Arguments

| Argument     | Description                          |
| ------------ | ------------------------------------ |
| `--dataset`  | Dataset to evaluate                  |
| `--seed`     | Seed ratio used for entity alignment |
| `--k_values` | Top-k candidate values to evaluate   |
| `--device`   | Computing device, e.g., `cuda:0`     |

---

## Examples

### FB15K-DB15K with 20% Seeds

```bash
python src/sparse_eiea.py --dataset FB15K-DB15K --seed 0.2
```

### FB15K-DB15K with 50% Seeds

```bash
python src/sparse_eiea.py --dataset FB15K-DB15K --seed 0.5
```

### FB15K-YAGO15K with 20% Seeds

```bash
python src/sparse_eiea.py --dataset FB15K-YAGO15K --seed 0.2
```

### FB15K-YAGO15K with 50% Seeds

```bash
python src/sparse_eiea.py --dataset FB15K-YAGO15K --seed 0.5
```

---

# Running Benchmarks

To evaluate multiple datasets, seed ratios, and top-k configurations:

```bash
python experiments/run_benchmark.py \
    --datasets FB15K-DB15K FB15K-YAGO15K \
    --seeds 0.2 0.5 0.8 \
    --k_values 50 100 200 500 1000
```

---

# Results

## SparseEIEA PERFORMANCE ON FB15K-DB15K AND FB15K-YAGO15K

<div align="center">
<img src="picture/result.png" width="100%" alt="SparseEIEA result"/>
</div>

## SparseEIEA Performance

| Dataset       | Seed |    k |     Hits@1 | Hits@10 |   MRR | Total Peak (MB) | Memory Reduction |
| ------------- | ---: | ---: | ---------: | ------: | ----: | --------------: | ---------------: |
| FB15K-DB15K   |  0.5 |  200 | **91.83%** |  95.87% | 0.935 |      **158.00** |       **83.34%** |
| FB15K-YAGO15K |  0.5 |  500 | **94.37%** |  97.34% | 0.955 |      **120.00** |       **83.34%** |
| FB15K-DB15K   |  0.8 | 1000 | **83.30%** |  92.21% | 0.865 |       **41.99** |       **79.60%** |
| FB15K-YAGO15K |  0.8 |  500 | **95.31%** |  98.62% | 0.962 |       **22.37** |       **90.90%** |

---

## Memory Scalability

SparseEIEA significantly improves scalability by replacing dense \(O(N^2)\) matrix storage with sparse \(O(N \cdot k)\) candidate representations.

| Entities (N) | Dense EIEA (Full Matrix) | SparseEIEA (k=50) | Memory Reduction |
| -----------: | -----------------------: | ----------------: | ---------------: |
|       15,000 |                809.20 MB |         158.00 MB |           80.48% |
|       50,000 |                  8.99 GB |            540 MB |           94.14% |
|      100,000 |                 35.96 GB |           1.08 GB |           97.00% |
|      500,000 |                898.89 GB |           5.40 GB |           99.40% |
|    1,000,000 |              **3.60 TB** |       **10.8 GB** |       **99.71%** |

---

## Key Findings

* **Memory Efficiency:** SparseEIEA achieves up to **83.34% memory reduction** in the evaluated entity alignment settings.
* **Accuracy Preservation:** More than **98% of dense alignment performance** is preserved.
* **Scalability:** SparseEIEA scales to **1 million entities using 10.8 GB**, compared with approximately **3.6 TB** for dense approaches.
* **Best Trade-off:** Values between **k = 200 and k = 500** generally provide an effective balance between alignment accuracy and memory consumption.

---

# SparseEIEA on DBP15K

SparseEIEA also includes support for experiments on DBP15K datasets.

```bash
cd DBP15K_src

python run.py zh_en
python run.py ja_en
python run.py fr_en
```

> **Note:** DBP15K support is currently under development and may be further improved in future versions.

---

# Output

Experimental results are automatically saved in the following format:

```text
results/sparse_eiea_{dataset}_{seed}_{timestamp}.txt
```

Each result file contains:

* Dense baseline performance
* Sparse results for different top-k values
* Hits@1
* Hits@10
* Mean Reciprocal Rank (MRR)
* Memory consumption
* Number of non-zero elements (NNZ)
* Matrix density
* Accuracy-memory trade-off analysis

---

# Citation

If you use SparseEIEA in your research, please cite:

```bibtex
@article{hasif2026sparseeiea,
  title={SparseEIEA: Memory-Efficient Multi-Modal Entity Alignment via Top-k Sparsification},
  author={Hasif Mahmud Mahim and Tiexin Wang},
  journal={arXiv preprint},
  year={2026}
}
```

---

# Contact

* **Hasif Mahmud Mahim** — [hasifmahmud98@gmail.com](mailto:hasifmahmud98@gmail.com)
* **Tiexin Wang** — [tiexin.wang@nuaa.edu.cn](mailto:tiexin.wang@nuaa.edu.cn)

---

# Acknowledgments

This work builds upon the **EIEA framework** and related research in multi-modal entity alignment. We sincerely thank the original authors for making their code and datasets publicly available.

**Reference:**

> L. Wang, C. Zhou, and B. Qin, "Explicit-Implicit Entity Alignment Method in Multi-modal Knowledge Graphs," in *Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD '25)*, 2025.
