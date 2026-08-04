# main_real.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
from config import CONFIG
from real_data_loader import load_real_dataset
from lsh_utils import LSHIndex, compute_dense_cs, compute_sparse_cs, rank_and_evaluate
import warnings
warnings.filterwarnings("ignore")

def run_real_experiment(data_dir, dataset_name):
    print("=" * 60)
    print(f"SPARSE EIEA: REAL DATASET EXPERIMENT - {dataset_name}")
    print("=" * 60)
    
    # 1. Load Real Data with BERT + Structure
    print(f"\n[1] Loading {dataset_name} dataset...")
    X1, X2, ground_truth, train_entities, test_entities = load_real_dataset(
        data_dir, dataset_name, 
        use_bert=True, 
        use_structure=CONFIG.get('use_structure', True)
    )
    
    # 2. Run DENSE baseline
    print(f"\n[2] Running Dense Baseline...")
    try:
        dense_cs, dense_stats = compute_dense_cs(X1, X2)
        dense_metrics = rank_and_evaluate(dense_cs, test_entities, ground_truth, is_sparse=False)
        
        print(f"    Memory: {dense_stats['memory_mb']:.2f} MB | Time: {dense_stats['time']:.3f}s")
        print(f"    Hits@1: {dense_metrics['Hits@1']:.4f} | MRR: {dense_metrics['MRR']:.4f}")
        print(f"    Test entities: {dense_metrics['n_test']}")
        
        results = [{
            'Dataset': dataset_name,
            'Method': 'Dense (Original)',
            'Top-K': 'N/A',
            'Hits@1': dense_metrics['Hits@1'],
            'MRR': dense_metrics['MRR'],
            'Memory (MB)': dense_stats['memory_mb'],
            'Time (s)': dense_stats['time'],
            'Density (%)': 100.0,
            'n_test': dense_metrics['n_test']
        }]
        
    except MemoryError:
        print("    DENSE BASELINE FAILED: Out of Memory!")
        results = []
    
    # 3. Build LSH Index
    print(f"\n[3] Building LSH Index...")
    lsh_index = LSHIndex(
        dim=CONFIG['embed_dim'],
        num_bits=CONFIG['lsh_bits'],
        num_tables=CONFIG['lsh_tables']
    )
    lsh_index.fit(X2)
    
    # 4. Run SPARSE methods
    print(f"\n[4] Running Sparse LSH with varying Top-K...")
    for k in CONFIG['top_k_list']:
        print(f"\n    Top-K = {k}...")
        sparse_cs, sparse_stats = compute_sparse_cs(X1, X2, lsh_index, top_k=k)
        sparse_metrics = rank_and_evaluate(sparse_cs, test_entities, ground_truth, is_sparse=True)
        
        print(f"        Memory: {sparse_stats['memory_mb']:.2f} MB | Time: {sparse_stats['time']:.3f}s")
        print(f"        Hits@1: {sparse_metrics['Hits@1']:.4f} | MRR: {sparse_metrics['MRR']:.4f}")
        print(f"        Density: {sparse_stats['density']*100:.2f}%")
        
        results.append({
            'Dataset': dataset_name,
            'Method': 'Sparse LSH',
            'Top-K': k,
            'Hits@1': sparse_metrics['Hits@1'],
            'MRR': sparse_metrics['MRR'],
            'Memory (MB)': sparse_stats['memory_mb'],
            'Time (s)': sparse_stats['time'],
            'Density (%)': sparse_stats['density'] * 100,
            'n_test': sparse_metrics['n_test']
        })
    
    # 5. Save results
    df = pd.DataFrame(results)
    print("\n" + "=" * 60)
    print(f"RESULTS FOR {dataset_name}")
    print("=" * 60)
    print(df.to_string(index=False))
    
    os.makedirs('results', exist_ok=True)
    df.to_csv(f'results/real_{dataset_name}.csv', index=False)
    
    # 6. Generate plot
    generate_plot(df, dataset_name)
    
    return df


def generate_plot(df, dataset_name):
    if len(df) == 0:
        return
    
    sparse_df = df[df['Method'] == 'Sparse LSH']
    dense_row = df[df['Method'] == 'Dense (Original)']
    
    if len(sparse_df) == 0:
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Plot 1: Accuracy vs Top-K
    ax1 = axes[0]
    ax1.plot(sparse_df['Top-K'], sparse_df['Hits@1'], 'o-', label='Sparse LSH')
    if len(dense_row) > 0:
        ax1.axhline(y=dense_row['Hits@1'].iloc[0], color='r', linestyle='--', label='Dense Baseline')
    ax1.set_xlabel('Top-K Candidates')
    ax1.set_ylabel('Hits@1')
    ax1.set_title(f'{dataset_name}: Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Plot 2: Memory vs Top-K
    ax2 = axes[1]
    ax2.plot(sparse_df['Top-K'], sparse_df['Memory (MB)'], 's-', color='green', label='Sparse LSH')
    if len(dense_row) > 0:
        ax2.axhline(y=dense_row['Memory (MB)'].iloc[0], color='r', linestyle='--', label='Dense Baseline')
    ax2.set_xlabel('Top-K Candidates')
    ax2.set_ylabel('Memory (MB)')
    ax2.set_title('Memory Usage')
    ax2.legend()
    ax2.grid(True)
    
    # Plot 3: Time vs Top-K
    ax3 = axes[2]
    ax3.plot(sparse_df['Top-K'], sparse_df['Time (s)'], '^-', color='orange', label='Sparse LSH')
    if len(dense_row) > 0:
        ax3.axhline(y=dense_row['Time (s)'].iloc[0], color='r', linestyle='--', label='Dense Baseline')
    ax3.set_xlabel('Top-K Candidates')
    ax3.set_ylabel('Time (seconds)')
    ax3.set_title('Computation Time')
    ax3.legend()
    ax3.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'results/real_{dataset_name}_tradeoff.png', dpi=150)
    print(f"    Plot saved to results/real_{dataset_name}_tradeoff.png")


if __name__ == "__main__":
    import traceback
    DATA_DIR = "./data/mmkg"
    
    # Available datasets
    datasets = ['zh_en', 'ja_en', 'fr_en', 'FBDB15K', 'FBYG15K']
    
    for dataset in datasets:
        try:
            df = run_real_experiment(DATA_DIR, dataset)
            print(f"\n✅ Completed {dataset}")
        except Exception as e:
            print(f"\n❌ Failed on {dataset}:")
            traceback.print_exc()