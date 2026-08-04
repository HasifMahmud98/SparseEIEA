# lsh_utils.py
import numpy as np
import scipy.sparse as sp
import faiss
import time

class LSHIndex:
    def __init__(self, dim, num_bits=32, num_tables=4):
        self.dim = dim
        self.num_bits = num_bits
        self.num_tables = num_tables
        self.indices = []
    
    def fit(self, embeddings):
        # Clean data
        if np.isnan(embeddings).any():
            embeddings = np.nan_to_num(embeddings)
        if np.isinf(embeddings).any():
            embeddings = np.clip(embeddings, -10, 10)
        
        embeddings = embeddings.astype('float32')
        
        if embeddings.shape[1] != self.dim:
            print(f"Warning: Resizing embeddings from {embeddings.shape[1]} to {self.dim}")
            # Pad or truncate
            if embeddings.shape[1] < self.dim:
                pad = np.zeros((embeddings.shape[0], self.dim - embeddings.shape[1]))
                embeddings = np.concatenate([embeddings, pad], axis=1)
            else:
                embeddings = embeddings[:, :self.dim]
        
        for _ in range(self.num_tables):
            index = faiss.IndexLSH(self.dim, self.num_bits)
            index.add(embeddings)
            self.indices.append(index)
        return self
    
    def query(self, query_embeddings, k=100):
        if np.isnan(query_embeddings).any():
            query_embeddings = np.nan_to_num(query_embeddings)
        query_embeddings = query_embeddings.astype('float32')
        
        # Handle dimension mismatch
        if query_embeddings.shape[1] != self.dim:
            if query_embeddings.shape[1] < self.dim:
                pad = np.zeros((query_embeddings.shape[0], self.dim - query_embeddings.shape[1]))
                query_embeddings = np.concatenate([query_embeddings, pad], axis=1)
            else:
                query_embeddings = query_embeddings[:, :self.dim]
        
        all_candidates = []
        for idx in self.indices:
            _, indices = idx.search(query_embeddings, k)
            all_candidates.append(indices)
        
        merged = []
        for i in range(len(query_embeddings)):
            cand_set = set()
            for table in all_candidates:
                cand_set.update(table[i])
            merged.append(list(cand_set)[:k])
        return merged


def compute_dense_cs(X1, X2):
    start = time.time()
    cs = X1 @ X2.T
    elapsed = time.time() - start
    mem = cs.nbytes / (1024 * 1024)
    return cs, {'time': elapsed, 'memory_mb': mem, 'type': 'dense'}


def compute_sparse_cs(X1, X2, lsh_index, top_k=100):
    start = time.time()
    N1 = X1.shape[0]
    N2 = X2.shape[0]
    
    candidates = lsh_index.query(X1, k=top_k)
    
    rows, cols, vals = [], [], []
    
    for i, cand_list in enumerate(candidates):
        if not cand_list or len(cand_list) == 0:
            cand_list = np.random.choice(N2, min(top_k, N2), replace=False).tolist()
        
        valid_cands = [j for j in cand_list if j < N2]
        if not valid_cands:
            valid_cands = [0]
        
        for j in valid_cands[:top_k]:
            score = float(np.dot(X1[i], X2[j]))
            if score > 1e-6:
                rows.append(i)
                cols.append(j)
                vals.append(score)
    
    if len(vals) == 0:
        # Fallback: use top-k random matches with small scores
        n_fallback = min(top_k, N1, N2)
        rows = np.random.choice(N1, n_fallback)
        cols = np.random.choice(N2, n_fallback)
        vals = np.ones(n_fallback) * 0.001
    
    sparse_cs = sp.csr_matrix((vals, (rows, cols)), shape=(N1, N2))
    elapsed = time.time() - start
    
    mem = (sparse_cs.data.nbytes + sparse_cs.indptr.nbytes + sparse_cs.indices.nbytes) / (1024 * 1024)
    
    return sparse_cs, {
        'time': elapsed,
        'memory_mb': mem,
        'nnz': sparse_cs.nnz,
        'density': sparse_cs.nnz / (N1 * N2) if N1 > 0 and N2 > 0 else 0,
        'type': 'sparse'
    }


def rank_and_evaluate(cs_matrix, test_pairs, ground_truth, is_sparse=False):
    if len(test_pairs) == 0:
        return {'Hits@1': 0.0, 'MRR': 0.0, 'n_test': 0}
    
    if is_sparse:
        dense_scores = cs_matrix.toarray()
    else:
        dense_scores = cs_matrix
    
    hits1 = 0
    mrr_sum = 0.0
    valid_count = 0
    
    for src_idx in test_pairs:
        if src_idx not in ground_truth:
            continue
        
        true_tgt = ground_truth[src_idx]
        
        if true_tgt >= dense_scores.shape[1]:
            continue
        
        scores = dense_scores[src_idx]
        sorted_indices = np.argsort(-scores)
        
        rank_positions = np.where(sorted_indices == true_tgt)[0]
        if len(rank_positions) == 0:
            continue
        
        rank_pos = rank_positions[0] + 1
        
        if rank_pos <= 1:
            hits1 += 1
        mrr_sum += 1.0 / rank_pos
        valid_count += 1
    
    if valid_count == 0:
        return {'Hits@1': 0.0, 'MRR': 0.0, 'n_test': 0}
    
    return {
        'Hits@1': hits1 / valid_count,
        'MRR': mrr_sum / valid_count,
        'n_test': valid_count
    }