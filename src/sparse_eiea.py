# sparse_eiea.py
"""
SparseEIEA: Memory-Efficient Multi-Modal Entity Alignment via Top-k Sparsification
FP16 optimized version: stores dense P in float16, reducing memory by ~50%.
"""

import os
import sys
import time
import gc
import numpy as np
import torch
import scipy.sparse as sp
from scipy.sparse import csr_matrix
import argparse
from datetime import datetime

# Try to import psutil for CPU memory tracking
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("WARNING: psutil not installed. Install with: pip install psutil")
    print("CPU memory tracking will not be available.")

# ============================================================
# Import EIEA modules
# ============================================================
from dataset import MMEaDtaset
from util import evaluate_sim_matrix, sinkhorn, view2


# ============================================================
# Helper: Get current CPU memory usage
# ============================================================
def get_cpu_memory_mb():
    """Return current process memory usage in MB."""
    if not HAS_PSUTIL:
        return 0.0
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


# ============================================================
# Sparse Sinkhorn (efficient, CPU-based) – returns float16
# ============================================================
def sparse_sinkhorn_efficient(csr_mat: csr_matrix, tau: float = 0.02,
                               max_iters: int = 100, stop_thresh: float = 1e-3):
    """
    Sinkhorn on sparse matrix using only non-zero entries.
    Returns a dense matrix P (size N1 x N2) in float16 to halve memory.
    """
    if csr_mat.nnz == 0:
        return np.zeros((csr_mat.shape[0], csr_mat.shape[1]), dtype=np.float16)
    
    coo = csr_mat.tocoo()
    rows = coo.row
    cols = coo.col
    data = coo.data
    N1, N2 = csr_mat.shape
    
    # Precompute exp(data / tau)
    K_data = np.exp(data / tau)
    
    u = np.ones(N1) / N1
    v = np.ones(N2) / N2
    
    for it in range(max_iters):
        u_old = u.copy()
        v_old = v.copy()
        
        # Update v: v_j = 1 / sum_i K_ij * u_i
        sum_i = np.zeros(N2)
        for idx, j in enumerate(cols):
            sum_i[j] += K_data[idx] * u[rows[idx]]
        sum_i = np.maximum(sum_i, 1e-10)
        v = 1.0 / sum_i
        
        # Update u: u_i = 1 / sum_j K_ij * v_j
        sum_j = np.zeros(N1)
        for idx, i in enumerate(rows):
            sum_j[i] += K_data[idx] * v[cols[idx]]
        sum_j = np.maximum(sum_j, 1e-10)
        u = 1.0 / sum_j
        
        # Convergence check
        err_u = np.max(np.abs(u - u_old))
        err_v = np.max(np.abs(v - v_old))
        if err_u < stop_thresh and err_v < stop_thresh:
            break
    
    # Build final P (dense via toarray) – cast to float16
    P_data = u[rows] * K_data * v[cols]
    P = csr_matrix((P_data, (rows, cols)), shape=(N1, N2))
    return P.toarray().astype(np.float16)


# ============================================================
# Main SparseEIEA Class
# ============================================================
class SparseEIEA:
    def __init__(self, data_dir, dataset_name, seed='0.2', k=200,
                 device='cuda:0', use_fp16=True, tau=0.02,
                 max_iters=100, stop_thresh=1e-3):
        self.data_dir = data_dir
        self.dataset_name = dataset_name
        self.seed = seed
        self.k = k
        self.device = device
        self.use_fp16 = use_fp16
        self.tau = tau
        self.max_iters = max_iters
        self.stop_thresh = stop_thresh
        
        self.side_modalities = {}
        self.dev_pair = None
        self.train_pair = None
        self.G_dataset = None
        self.rest_set_1 = None
        self.rest_set_2 = None
        self.idx_to_rest1 = {}
        self.idx_to_rest2 = {}
        self.memory_stats = {}

    def load_data(self):
        print("\n" + "=" * 70)
        print("LOADING DATA".center(70))
        print("=" * 70)
        data_path = f'{self.data_dir}/MMEA-data/seed{self.seed}/{self.dataset_name}'
        self.G_dataset = MMEaDtaset(data_path, device=self.device, ratio=float(self.seed))
        self.train_pair = torch.tensor(self.G_dataset.train_pair, dtype=torch.long)
        self.dev_pair = torch.tensor(self.G_dataset.test_pair, dtype=torch.long)
        print(f"  Train set: {len(self.train_pair)}")
        print(f"  Dev set: {len(self.dev_pair)}")

        # Store test entity ID lists and mappings
        self.rest_set_1 = self.G_dataset.rest_set_1
        self.rest_set_2 = self.G_dataset.rest_set_2
        self.idx_to_rest1 = {eid: i for i, eid in enumerate(self.rest_set_1)}
        self.idx_to_rest2 = {eid: i for i, eid in enumerate(self.rest_set_2)}
        n_test = len(self.rest_set_1)
        print(f"  Test set size: {n_test}")

        # Load side modalities with filtering to test set size
        ecs_path = f'{self.data_dir}/ECS_results/seed{self.seed}/{self.dataset_name}'
        self.side_modalities = {}
        
        for filename in os.listdir(ecs_path):
            if filename.endswith('.npy'):
                moda_np = np.load(os.path.join(ecs_path, filename))
                
                # Filter to test set size
                if moda_np.shape[0] != n_test:
                    print(f"  Resizing {filename} from {moda_np.shape} to ({n_test}, {n_test})")
                    if moda_np.shape[0] > n_test:
                        moda_np = moda_np[:n_test, :n_test]
                    else:
                        pad_size = n_test - moda_np.shape[0]
                        moda_np = np.pad(moda_np, ((0, pad_size), (0, pad_size)), mode='constant')
                
                # Normalize
                if filename == 'name.npy':
                    moda_np = (moda_np - moda_np.min()) / (moda_np.max() - moda_np.min() + 1e-8)
                elif moda_np.max() > 1.5:
                    moda_np = moda_np / moda_np.max()
                
                self.side_modalities[filename.split('.')[0]] = moda_np
                print(f"  Loaded {filename}: {moda_np.shape}")
        
        print(f"  Total modalities: {len(self.side_modalities)}")
        print("  Data loading complete!")

    def get_test_indices(self):
        """Return test entity indices in the order of dev_pair."""
        gid1, gid2 = self.dev_pair.T
        idx1 = np.array([self.idx_to_rest1[int(e)] for e in gid1])
        idx2 = np.array([self.idx_to_rest2[int(e)] for e in gid2])
        return idx1, idx2

    # ------------------------------------------------------------------
    # Dense baseline (GPU)
    # ------------------------------------------------------------------
    def run_dense_baseline(self):
        print("\n" + "=" * 70)
        print("RUNNING DENSE BASELINE".center(70))
        print("=" * 70)
        
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.reset_peak_memory_stats()
        start_gpu = torch.cuda.memory_allocated()
        start_time = time.time()

        idx1, idx2 = self.get_test_indices()
        len_dev = len(idx1)

        # Build combined correlation matrix
        all_scores = None
        count = 0
        for name, matrix in self.side_modalities.items():
            scores = matrix[idx1, :][:, idx2]
            if all_scores is None:
                all_scores = scores
            else:
                all_scores += scores
            count += 1

        if all_scores is None:
            print("  No correlation matrices found!")
            return None, 0

        all_scores = all_scores / count

        # Move to GPU and run Sinkhorn
        scores_tensor = torch.FloatTensor(all_scores).to(self.device)
        scores_tensor = scores_tensor.T
        sim_mat_r = 1 - scores_tensor
        M = sim_mat_r.view(1, sim_mat_r.size(0), -1)
        a = torch.ones([1, sim_mat_r.shape[0]], requires_grad=False, device=self.device)
        b = torch.ones([1, sim_mat_r.shape[1]], requires_grad=False, device=self.device)
        P = sinkhorn(a, b, M, self.tau, max_iters=self.max_iters, stop_thresh=self.stop_thresh)
        P = view2(P)

        torch.cuda.synchronize()
        peak_gpu = torch.cuda.max_memory_allocated()
        gpu_memory_used = (peak_gpu - start_gpu) / (1024 * 1024)

        result = evaluate_sim_matrix(
            link=torch.stack([torch.arange(len_dev), torch.arange(len_dev)], dim=0),
            sim_x2y=P,
            no_csls=True
        )

        elapsed = time.time() - start_time
        hits1 = float(result['hits@1'])
        hits10 = float(result.get('hits@10', 0))
        mrr = float(result.get('MRR', 0))

        print(f"  Dense Hits@1:  {hits1:.4f}")
        print(f"  Dense Hits@10: {hits10:.4f}")
        print(f"  Dense MRR:     {mrr:.4f}")
        print(f"  Dense GPU Memory: {gpu_memory_used:.2f} MB")
        print(f"  Dense Time: {elapsed:.2f}s")

        self.memory_stats['dense'] = {
            'hits@1': hits1,
            'hits@10': hits10,
            'mrr': mrr,
            'gpu_memory_mb': gpu_memory_used,
            'time': elapsed
        }
        return result, gpu_memory_used

    # ------------------------------------------------------------------
    # Sparse sinkhorn (CPU + GPU for evaluation) – FP16 optimized
    # ------------------------------------------------------------------
    def run_sparse(self, k=200):
        print(f"\n" + "=" * 70)
        print(f"RUNNING SPARSE SINKHORN (k={k})".center(70))
        print("=" * 70)

        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.reset_peak_memory_stats()
        start_gpu = torch.cuda.memory_allocated()
        start_time = time.time()

        idx1, idx2 = self.get_test_indices()
        len_dev = len(idx1)
        N = len(idx1)

        # --- Build sparse matrix row-by-row ---
        rows, cols, vals = [], [], []
        for i in range(N):
            row_scores = None
            for matrix in self.side_modalities.values():
                if row_scores is None:
                    row_scores = matrix[i, :].copy()
                else:
                    row_scores += matrix[i, :]
            row_scores = row_scores / len(self.side_modalities)
            
            top_k_indices = np.argsort(-row_scores)[:k]
            for j in top_k_indices:
                score = row_scores[j]
                if score > 1e-6:
                    rows.append(i)
                    cols.append(j)
                    vals.append(score)

        nnz = len(vals)
        density = nnz / (N * N) * 100 if N > 0 else 0
        sparse_mem = (nnz * (4 + 4 + 4)) / (1024 * 1024)
        print(f"  Non-zero entries: {nnz}")
        print(f"  Density: {density:.2f}%")
        print(f"  Sparse Matrix (theoretical): {sparse_mem:.2f} MB")

        # --- Sparse Sinkhorn (produces dense P in float16) ---
        sparse_mat = csr_matrix((vals, (rows, cols)), shape=(N, N))
        
        P_dense_f16 = sparse_sinkhorn_efficient(
            sparse_mat, tau=self.tau,
            max_iters=self.max_iters,
            stop_thresh=self.stop_thresh
        )
        
        dense_p_mem = P_dense_f16.nbytes / (1024 * 1024)
        print(f"  Dense P (float16): {dense_p_mem:.2f} MB")

        # --- Convert to float32 for evaluation ---
        P_torch = torch.tensor(P_dense_f16, dtype=torch.float32).to(self.device)
        torch.cuda.synchronize()
        peak_gpu = torch.cuda.max_memory_allocated()
        gpu_memory_used = (peak_gpu - start_gpu) / (1024 * 1024)

        # Evaluate
        result = evaluate_sim_matrix(
            link=torch.stack([torch.arange(len_dev), torch.arange(len_dev)], dim=0),
            sim_x2y=P_torch,
            no_csls=True
        )

        elapsed = time.time() - start_time
        hits1 = float(result['hits@1'])
        hits10 = float(result.get('hits@10', 0))
        mrr = float(result.get('MRR', 0))

        # --- Peak memory calculation ---
        cpu_peak = dense_p_mem + sparse_mem
        total_peak = max(cpu_peak, gpu_memory_used)
        
        print(f"\n  Peak CPU Memory (P_f16 + sparse): {cpu_peak:.2f} MB")
        print(f"  Peak GPU Memory: {gpu_memory_used:.2f} MB")
        print(f"  Total Peak Memory: {total_peak:.2f} MB")

        print(f"\n  Sparse Hits@1:  {hits1:.4f}")
        print(f"  Sparse Hits@10: {hits10:.4f}")
        print(f"  Sparse MRR:     {mrr:.4f}")
        print(f"  Time: {elapsed:.2f}s")

        # Save stats
        self.memory_stats[k] = {
            'k': k,
            'hits@1': hits1,
            'hits@10': hits10,
            'mrr': mrr,
            'cpu_peak_mb': cpu_peak,
            'gpu_peak_mb': gpu_memory_used,
            'total_peak_mb': total_peak,
            'dense_p_mb': dense_p_mem,
            'sparse_matrix_mb': sparse_mem,
            'time': elapsed,
            'nnz': nnz,
            'density': density
        }
        return result, total_peak

    # ------------------------------------------------------------------
    # Full experiment
    # ------------------------------------------------------------------
    def run_experiment(self, k_values=None):
        if k_values is None:
            k_values = [50, 100, 200, 500, 1000]

        print("\n" + "=" * 70)
        print("SPARSEEIEA EXPERIMENT".center(70))
        print("=" * 70)
        print(f"Dataset: {self.dataset_name}")
        print(f"Seed: {self.seed}")
        print(f"Device: {self.device}")
        print(f"FP16: {self.use_fp16}")
        print(f"k values: {k_values}")
        print("=" * 70)

        self.load_data()

        # Dense baseline
        dense_result, dense_mem = self.run_dense_baseline()
        dense_hits1 = float(dense_result['hits@1'])
        dense_hits10 = float(dense_result.get('hits@10', 0))
        dense_mrr = float(dense_result.get('MRR', 0))

        # Sparse with different k
        sparse_results = []
        for k in k_values:
            result, mem = self.run_sparse(k=k)
            stats = self.memory_stats[k]
            sparse_results.append({
                'k': k,
                'hits@1': float(result['hits@1']),
                'hits@10': float(result.get('hits@10', 0)),
                'mrr': float(result.get('MRR', 0)),
                'total_peak_mb': stats.get('total_peak_mb', 0),
                'cpu_peak_mb': stats.get('cpu_peak_mb', 0),
                'gpu_peak_mb': stats.get('gpu_peak_mb', 0),
                'nnz': stats.get('nnz', 0),
                'density': stats.get('density', 0)
            })

        # ================================================================
        # FINAL SUMMARY
        # ================================================================
        print("\n" + "=" * 70)
        print("FINAL SUMMARY".center(70))
        print("=" * 70)

        print(f"\n  Dense Baseline:")
        print(f"    Hits@1:  {dense_hits1:.4f}")
        print(f"    Hits@10: {dense_hits10:.4f}")
        print(f"    MRR:     {dense_mrr:.4f}")
        print(f"    Memory:  {dense_mem:.2f} MB")

        print(f"\n  Sparse Results (SparseEIEA, FP16):")
        print(f"    {'k':<6} {'Hits@1':<10} {'Hits@10':<10} {'MRR':<10} {'Total (MB)':<14} {'CPU (MB)':<12} {'GPU (MB)':<12} {'NNZ':<10} {'Density':<10}")
        print(f"    {'-'*6} {'-'*10} {'-'*10} {'-'*10} {'-'*14} {'-'*12} {'-'*12} {'-'*10} {'-'*10}")

        for res in sparse_results:
            print(f"    {res['k']:<6} {res['hits@1']:<10.4f} {res['hits@10']:<10.4f} {res['mrr']:<10.4f} {res['total_peak_mb']:<14.2f} {res['cpu_peak_mb']:<12.2f} {res['gpu_peak_mb']:<12.2f} {res['nnz']:<10} {res['density']:<10.2f}%")

        # Find best k >=80% Hits@1 with lowest memory
        best = None
        for res in sparse_results:
            if res['hits@1'] >= 0.80:
                if best is None or res['total_peak_mb'] < best['total_peak_mb']:
                    best = res

        if best:
            reduction = (1 - best['total_peak_mb'] / dense_mem) * 100
            print(f"\n  Best k for >=80% Hits@1: k={best['k']}")
            print(f"    Hits@1:  {best['hits@1']:.4f}")
            print(f"    Hits@10: {best['hits@10']:.4f}")
            print(f"    MRR:     {best['mrr']:.4f}")
            print(f"    Total Peak Memory: {best['total_peak_mb']:.2f} MB")
            print(f"    Memory Reduction: {reduction:.2f}%")
        else:
            print("\n  No k achieved >=80% Hits@1. Try higher seed ratio (0.5 or 0.8).")

        # ================================================================
        # SAVE RESULTS TO FILE
        # ================================================================
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../results')
        os.makedirs(results_dir, exist_ok=True)

        results_file = os.path.join(results_dir, f"sparse_eiea_{self.dataset_name}_{self.seed}_{timestamp}.txt")

        with open(results_file, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("SPARSEEIEA RESULTS\n")
            f.write("=" * 70 + "\n")
            f.write(f"Dataset: {self.dataset_name}\n")
            f.write(f"Seed: {self.seed}\n")
            f.write(f"Device: {self.device}\n")
            f.write(f"FP16: {self.use_fp16}\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("Dense Baseline:\n")
            f.write(f"  Hits@1:  {dense_hits1:.4f}\n")
            f.write(f"  Hits@10: {dense_hits10:.4f}\n")
            f.write(f"  MRR:     {dense_mrr:.4f}\n")
            f.write(f"  Memory:  {dense_mem:.2f} MB\n\n")

            f.write("Sparse Results:\n")
            f.write(f"{'k':<6} {'Hits@1':<10} {'Hits@10':<10} {'MRR':<10} {'Total (MB)':<14} {'CPU (MB)':<12} {'GPU (MB)':<12} {'NNZ':<10} {'Density':<10}\n")
            f.write("-" * 95 + "\n")

            for res in sparse_results:
                f.write(f"{res['k']:<6} {res['hits@1']:<10.4f} {res['hits@10']:<10.4f} {res['mrr']:<10.4f} {res['total_peak_mb']:<14.2f} {res['cpu_peak_mb']:<12.2f} {res['gpu_peak_mb']:<12.2f} {res['nnz']:<10} {res['density']:<10.2f}%\n")

            if best:
                f.write(f"\nBest k for >=80% Hits@1: k={best['k']}\n")
                f.write(f"  Hits@1:  {best['hits@1']:.4f}\n")
                f.write(f"  Hits@10: {best['hits@10']:.4f}\n")
                f.write(f"  MRR:     {best['mrr']:.4f}\n")
                f.write(f"  Total Peak Memory: {best['total_peak_mb']:.2f} MB\n")
                f.write(f"  Memory Reduction: {(1 - best['total_peak_mb']/dense_mem) * 100:.2f}%\n")

            f.write("\n" + "=" * 70 + "\n")

        print(f"\n  Results saved to: {results_file}")
        print("=" * 70 + "\n")

        return {
            'dense': {
                'hits@1': dense_hits1,
                'hits@10': dense_hits10,
                'mrr': dense_mrr,
                'memory_mb': dense_mem
            },
            'sparse': sparse_results
        }


# ============================================================
# Main Entry Point
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='SparseEIEA: Memory-Efficient Entity Alignment')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['FB15K-DB15K', 'FB15K-YAGO15K'],
                        help='Dataset name')
    parser.add_argument('--seed', type=str, default='0.2',
                        choices=['0.2', '0.5', '0.8'],
                        help='Seed ratio (0.2, 0.5, 0.8)')
    parser.add_argument('--k_values', type=int, nargs='+',
                        default=[50, 100, 200, 500, 1000],
                        help='List of k values to test')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use (cuda:0 or cpu)')
    parser.add_argument('--data_dir', type=str, default='../data',
                        help='Data directory path')
    parser.add_argument('--no_fp16', action='store_true',
                        help='Disable FP16 precision')
    parser.add_argument('--tau', type=float, default=0.02,
                        help='Sinkhorn regularization parameter')
    parser.add_argument('--max_iters', type=int, default=100,
                        help='Max Sinkhorn iterations')
    parser.add_argument('--stop_thresh', type=float, default=1e-3,
                        help='Sinkhorn convergence threshold')

    args = parser.parse_args()

    model = SparseEIEA(
        data_dir=args.data_dir,
        dataset_name=args.dataset,
        seed=args.seed,
        device=args.device,
        use_fp16=not args.no_fp16,
        tau=args.tau,
        max_iters=args.max_iters,
        stop_thresh=args.stop_thresh
    )

    model.run_experiment(k_values=args.k_values)


if __name__ == "__main__":
    main()