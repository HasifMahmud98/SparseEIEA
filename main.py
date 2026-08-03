# main.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from config import CONFIG
from data_utils import generate_synthetic_data
from lsh_utils import LSHIndex, compute_dense_cs, compute_sparse_cs, rank_and_evaluate
import warnings
warnings.filterwarnings("ignore")

def run_experiment():
    print("=" * 60)
    print("SPARSE EIEA EXPERIMENT: DENSE vs LSH")
    print("=" * 60)
    
    # 1. Generate Data
    print("\n[1] Generating Synthetic Knowledge Graph Embeddings...")
    X1, X2, ground_truth, train_entities, test_entities = generate_synthetic_data()
    N = CONFIG['num_entities']
    
    results = []
    
    # 2. Run DENSE baseline (O(N^2))
    print(f"\n[2] Running Dense Baseline (N={N})...")
    dense_cs, dense_stats = compute_dense_cs(X1, X2)
    dense_metrics = rank_and_evaluate(dense_cs, test_entities, ground_truth, is_sparse=False)
    
    print(f"    Memory: {dense_stats['memory_mb']:.2f} MB | Time: {dense_stats['time']:.3f}s")
    print(f"    Hits@1: {dense_metrics['Hits@1']:.4f} | MRR: {dense_metrics['MRR']:.4f}")
    
    results.append({
        'Method': 'Dense (Original)',
        'Top-K': 'N/A',
        'Hits@1': dense_metrics['Hits@1'],
        'MRR': dense_metrics['MRR'],
        'Memory (MB)': dense_stats['memory_mb'],
        'Time (s)': dense_stats['time'],
        'Density (%)': 100.0
    })
    
    # 3. Run SPARSE methods (O(N * k)) for varying top_k
    print("\n[3] Building LSH Index for Sparse Methods...")
    lsh_index = LSHIndex(dim=CONFIG['embed_dim'], 
                         num_bits=CONFIG['lsh_bits'], 
                         num_tables=CONFIG['lsh_tables'])
    lsh_index.fit(X2)
    
    top_k_values = CONFIG['top_k_list']
    print(f"    Testing Top-K values: {top_k_values}")
    
    for k in top_k_values:
        print(f"\n[4] Running Sparse LSH (Top-K = {k})...")
        sparse_cs, sparse_stats = compute_sparse_cs(X1, X2, lsh_index, top_k=k)
        sparse_metrics = rank_and_evaluate(sparse_cs, test_entities, ground_truth, is_sparse=True)
        
        print(f"    Memory: {sparse_stats['memory_mb']:.2f} MB | Time: {sparse_stats['time']:.3f}s | Density: {sparse_stats['density']*100:.2f}%")
        print(f"    Hits@1: {sparse_metrics['Hits@1']:.4f} | MRR: {sparse_metrics['MRR']:.4f}")
        
        results.append({
            'Method': 'Sparse LSH',
            'Top-K': k,
            'Hits@1': sparse_metrics['Hits@1'],
            'MRR': sparse_metrics['MRR'],
            'Memory (MB)': sparse_stats['memory_mb'],
            'Time (s)': sparse_stats['time'],
            'Density (%)': sparse_stats['density'] * 100
        })
    
    # 4. Create Results DataFrame
    df = pd.DataFrame(results)
    print("\n" + "=" * 60)
    print("FINAL RESULTS TABLE")
    print("=" * 60)
    print(df.to_string(index=False))
    
    # 5. Generate Plots
    print("\n[5] Generating Trade-off Plots...")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Plot 1: Accuracy vs Top-K
    ax1 = axes[0]
    sparse_df = df[df['Method'] == 'Sparse LSH']
    ax1.plot(sparse_df['Top-K'], sparse_df['Hits@1'], 'o-', label='Sparse LSH')
    ax1.axhline(y=dense_metrics['Hits@1'], color='r', linestyle='--', label='Dense Baseline')
    ax1.set_xlabel('Top-K Candidates')
    ax1.set_ylabel('Hits@1')
    ax1.set_title('Accuracy vs. Candidates')
    ax1.legend()
    ax1.grid(True)
    
    # Plot 2: Memory vs Top-K
    ax2 = axes[1]
    ax2.plot(sparse_df['Top-K'], sparse_df['Memory (MB)'], 's-', color='green', label='Sparse LSH')
    ax2.axhline(y=dense_stats['memory_mb'], color='r', linestyle='--', label='Dense Baseline')
    ax2.set_xlabel('Top-K Candidates')
    ax2.set_ylabel('Memory (MB)')
    ax2.set_title('Memory Usage')
    ax2.legend()
    ax2.grid(True)
    
    # Plot 3: Time vs Top-K
    ax3 = axes[2]
    ax3.plot(sparse_df['Top-K'], sparse_df['Time (s)'], '^-', color='orange', label='Sparse LSH')
    ax3.axhline(y=dense_stats['time'], color='r', linestyle='--', label='Dense Baseline')
    ax3.set_xlabel('Top-K Candidates')
    ax3.set_ylabel('Time (seconds)')
    ax3.set_title('Computation Time')
    ax3.legend()
    ax3.grid(True)
    
    plt.tight_layout()
    plt.savefig('results_tradeoff.png', dpi=150)
    print("    Plot saved as 'results_tradeoff.png'")
    
    # Save CSV
    df.to_csv('experiment_results.csv', index=False)
    print("    Results saved as 'experiment_results.csv'")
    
    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE!")
    print(f"Key Finding: Sparse LSH (k={sparse_df['Top-K'].iloc[2]}) uses {sparse_df['Memory (MB)'].iloc[2]/dense_stats['memory_mb']:.1%} memory, {sparse_df['Time (s)'].iloc[2]/dense_stats['time']:.1%} time, while maintaining {sparse_df['Hits@1'].iloc[2]/dense_metrics['Hits@1']:.1%} accuracy.")
    print("=" * 60)
    
    return df

if __name__ == "__main__":
    run_experiment()