# data_utils.py
import numpy as np
from config import CONFIG

def generate_synthetic_data():
    """
    Generates two KGs (source and target) with known alignments.
    - KG1: Random embeddings.
    - KG2: Noisy versions of KG1 for ALL entities.
    - This ensures every entity has a true match.
    """
    N = CONFIG['num_entities']
    dim = CONFIG['embed_dim']
    noise = CONFIG['noise_level']
    
    # Source KG: Random normal embeddings
    X1 = np.random.randn(N, dim).astype('float32')
    
    # Target KG: Start as noisy copies of KG1
    X2 = X1 + np.random.randn(N, dim).astype('float32') * noise
    
    # Normalize embeddings (improves LSH and dot product performance)
    X1 = X1 / np.linalg.norm(X1, axis=1, keepdims=True)
    X2 = X2 / np.linalg.norm(X2, axis=1, keepdims=True)
    
    # Ground truth: For each source entity i, its true match in target is i
    ground_truth = np.arange(N)
    
    # Split into train (20%) and test (80%) following EIEA protocol
    aligned_count = int(N * CONFIG['train_ratio'])
    train_entities = list(range(aligned_count))
    test_entities = list(range(aligned_count, N))
    
    print(f"Generated {N} entities per KG.")
    print(f"Training seeds: {len(train_entities)} (first {CONFIG['train_ratio']*100:.0f}%)")
    print(f"Test pairs: {len(test_entities)}")
    print(f"Noise level: {noise}")
    
    return X1, X2, ground_truth, train_entities, test_entities