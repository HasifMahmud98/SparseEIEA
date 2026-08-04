# real_data_loader.py
import os
import numpy as np
import pickle
from collections import defaultdict

try:
    from sentence_transformers import SentenceTransformer
    HAS_SBERT = True
except ImportError:
    HAS_SBERT = False
    print("WARNING: Install sentence-transformers")

try:
    import networkx as nx
    HAS_NX = True
except ImportError:
    HAS_NX = False
    print("WARNING: Install networkx for structural features")


def load_real_dataset(data_dir, dataset_name, use_bert=True, use_structure=True):
    """
    Load a real entity alignment dataset with BERT + structural features.
    """
    dataset_map = {
        'FBDB15K': ('FBDB15K/norm', 'FBDB15K'),
        'FBYG15K': ('FBYG15K/norm', 'FBYG15K'),
        'zh_en': ('DBP15K/zh_en', 'DBP15K'),
        'ja_en': ('DBP15K/ja_en', 'DBP15K'),
        'fr_en': ('DBP15K/fr_en', 'DBP15K'),
    }
    
    if dataset_name not in dataset_map:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    rel_path, base_name = dataset_map[dataset_name]
    full_path = os.path.join(data_dir, rel_path)
    
    print(f"Loading {dataset_name} from {full_path}...")
    
    # 1. Load entity IDs, names, and id->index mappings
    ent_ids_1, names_1, id_to_idx1 = load_entity_ids(os.path.join(full_path, 'ent_ids_1'))
    ent_ids_2, names_2, id_to_idx2 = load_entity_ids(os.path.join(full_path, 'ent_ids_2'))
    
    # 2. Load alignment pairs
    ill_ent_ids = load_alignment_pairs(os.path.join(full_path, 'ill_ent_ids'))
    
    # 3. Load triples for structural features
    triples_1 = load_triples(os.path.join(full_path, 'triples_1'))
    triples_2 = load_triples(os.path.join(full_path, 'triples_2'))
    
    # 4. Create ground truth mapping only for existing entity IDs
    ground_truth = {}
    for src_id, tgt_id in ill_ent_ids:
        if src_id in id_to_idx1 and tgt_id in id_to_idx2:
            ground_truth[src_id] = tgt_id
        else:
            if src_id not in id_to_idx1:
                print(f"Warning: Source entity {src_id} not found in KG1")
            if tgt_id not in id_to_idx2:
                print(f"Warning: Target entity {tgt_id} not found in KG2")
    
    # 5. Get name embeddings (in same order as ent_ids_1/ent_ids_2)
    if use_bert and HAS_SBERT:
        X1_name, X2_name = create_bert_embeddings(names_1, names_2)
    else:
        X1_name, X2_name = create_name_embeddings(names_1, names_2, dim=128)
    
    # 6. Get structural embeddings (in same order as ent_ids)
    if use_structure and HAS_NX:
        X1_struct, X2_struct = create_graph_embeddings(
            ent_ids_1, ent_ids_2, triples_1, triples_2, dim=64
        )
        print(f"Structure embedding dim: {X1_struct.shape[1]}")
    else:
        print("Skipping structural features")
        X1_struct = np.zeros((len(ent_ids_1), 64))
        X2_struct = np.zeros((len(ent_ids_2), 64))
    
    # 7. Concatenate features
    X1 = np.concatenate([X1_name, X1_struct], axis=1)
    X2 = np.concatenate([X2_name, X2_struct], axis=1)
    print(f"Combined embedding dim: {X1.shape[1]}")
    
    # 8. Filter embeddings to only aligned entities
    aligned_src = sorted(ground_truth.keys())
    aligned_tgt = sorted(set(ground_truth.values()))
    
    # Use id_to_idx mapping to get correct indices
    X1_filtered = np.array([X1[id_to_idx1[src]] for src in aligned_src])
    X2_filtered = np.array([X2[id_to_idx2[tgt]] for tgt in aligned_tgt])
    
    # 9. Create compact indexing for train/test split
    src_compact = {src: i for i, src in enumerate(aligned_src)}
    tgt_compact = {tgt: j for j, tgt in enumerate(aligned_tgt)}
    
    # 10. Re-index ground truth
    gt_compact = {}
    for src, tgt in ground_truth.items():
        if src in src_compact and tgt in tgt_compact:
            gt_compact[src_compact[src]] = tgt_compact[tgt]
    
    # 11. Split into train (20%) and test (80%)
    all_src_indices = sorted(gt_compact.keys())
    n_train = int(len(all_src_indices) * 0.2)
    train_entities = all_src_indices[:n_train]
    test_entities = all_src_indices[n_train:]
    
    print(f"Loaded {len(aligned_src)} aligned entities in KG1, {len(aligned_tgt)} in KG2")
    print(f"Total ground truth pairs: {len(gt_compact)}")
    print(f"Train: {len(train_entities)}, Test: {len(test_entities)}")
    print(f"Embedding shape: KG1 {X1_filtered.shape}, KG2 {X2_filtered.shape}")
    
    return X1_filtered, X2_filtered, gt_compact, train_entities, test_entities


def load_entity_ids(filepath):
    entities = []
    names = []
    id_to_idx = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            parts = line.strip().split()
            if len(parts) >= 2:
                eid = int(parts[0])
                uri = parts[1]
                name = uri.split('/')[-1].replace('_', ' ').replace('%20', ' ')
                entities.append(eid)
                names.append(name)
                id_to_idx[eid] = idx
    return entities, names, id_to_idx


def load_alignment_pairs(filepath):
    pairs = []
    if not os.path.exists(filepath):
        base_dir = os.path.dirname(filepath)
        for alt in ['sup_ent_ids', 'ref_ent_ids']:
            alt_path = os.path.join(base_dir, alt)
            if os.path.exists(alt_path):
                filepath = alt_path
                break
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                pairs.append((int(parts[0]), int(parts[1])))
    return pairs


def load_triples(filepath):
    triples = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    triples.append((int(parts[0]), int(parts[1]), int(parts[2])))
    except FileNotFoundError:
        print(f"Warning: Triples file {filepath} not found")
    return triples


def create_bert_embeddings(names_1, names_2):
    model = SentenceTransformer('all-mpnet-base-v2')
    print("Encoding names with BERT (all-mpnet-base-v2)...")
    X1 = model.encode(names_1, show_progress_bar=True, normalize_embeddings=True)
    X2 = model.encode(names_2, show_progress_bar=True, normalize_embeddings=True)
    return X1.astype('float32'), X2.astype('float32')


def create_name_embeddings(names_1, names_2, dim=128):
    all_names = names_1 + names_2
    vocab = {}
    for name in all_names:
        for char in name:
            if char not in vocab:
                vocab[char] = len(vocab)
    
    def name_to_emb(name):
        emb = np.zeros(dim)
        for i, char in enumerate(name):
            if char in vocab:
                idx = vocab[char] % dim
                emb[idx] += 1.0
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        return emb
    
    X1 = np.array([name_to_emb(name) for name in names_1]).astype('float32')
    X2 = np.array([name_to_emb(name) for name in names_2]).astype('float32')
    return X1, X2


def create_graph_embeddings(ent_ids_1, ent_ids_2, triples_1, triples_2, dim=64):
    """
    Create structural embeddings using networkx.
    """
    print("Building graph embeddings using networkx...")
    
    # Build graphs
    G1 = nx.Graph()
    G2 = nx.Graph()
    
    for h, r, t in triples_1:
        G1.add_edge(h, t)
    for h, r, t in triples_2:
        G2.add_edge(h, t)
    
    # Compute node features
    def get_node_features(G, entity_ids, dim):
        features = []
        
        # Precompute degree and clustering
        degrees = dict(G.degree())
        clustering = nx.clustering(G)
        
        # Use PageRank as additional feature
        try:
            pagerank = nx.pagerank(G, max_iter=100)
        except:
            pagerank = {node: 1.0/len(G) for node in G.nodes()}
        
        for eid in entity_ids:
            if eid not in G:
                feat = np.zeros(dim)
                features.append(feat)
                continue
            
            # Degree (normalized)
            deg = degrees.get(eid, 0)
            deg_norm = min(deg / 100.0, 1.0)
            
            # Clustering coefficient
            clust = clustering.get(eid, 0)
            
            # PageRank
            pr = pagerank.get(eid, 0.001)
            
            # Neighbor degree statistics
            neighbors = list(G.neighbors(eid))
            if neighbors:
                neighbor_degrees = [degrees.get(n, 0) for n in neighbors]
                mean_nd = min(np.mean(neighbor_degrees) / 100.0, 1.0)
                std_nd = min(np.std(neighbor_degrees) / 100.0, 1.0)
            else:
                mean_nd = 0
                std_nd = 0
            
            # Build feature vector
            feat = np.zeros(dim)
            feat[0] = deg_norm
            feat[1] = clust
            feat[2] = pr * 10  # Scale up
            feat[3] = mean_nd
            feat[4] = std_nd
            
            # Fill rest with random but deterministic values
            np.random.seed(eid % 1000)
            feat[5:min(20, dim)] = np.random.randn(min(15, dim-5)) * 0.05
            
            features.append(feat)
        
        return np.array(features)
    
    X1 = get_node_features(G1, ent_ids_1, dim)
    X2 = get_node_features(G2, ent_ids_2, dim)
    
    # Normalize features
    X1 = X1 / (np.linalg.norm(X1, axis=1, keepdims=True) + 1e-8)
    X2 = X2 / (np.linalg.norm(X2, axis=1, keepdims=True) + 1e-8)
    
    print(f"Graph embeddings created: {X1.shape}")
    
    return X1.astype('float32'), X2.astype('float32')