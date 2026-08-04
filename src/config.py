# config.py
# CONFIG = {
#     # Dataset generation
#     'num_entities': 5000,      # Start with 5k. Try 10k, 20k, 50k later.
#     'embed_dim': 128,          # Dimension of features
#     'noise_level': 0.01,       # Reduced noise for clearer alignment (0.01 = 1% noise)
    
#     # LSH Parameters
#     'top_k_list': [10, 50, 100, 200],  # Candidates to retrieve
#     'lsh_bits': 32,            # Accuracy of LSH
#     'lsh_tables': 4,           # More tables = better recall
    
#     # Evaluation
#     'train_ratio': 0.2,        # 20% alignment seeds
# }

# CONFIG = {
#     'num_entities': 15000,      # 15k entities
#     'embed_dim': 128,
#     'noise_level': 0.01,
#     'top_k_list': [10, 50, 100, 200],
#     'lsh_bits': 32,
#     'lsh_tables': 4,
#     'train_ratio': 0.2,
# }

# CONFIG = {
#     'embed_dim': 384,          # !!! Changed from 128 to 384 (BERT dimension)
#     'noise_level': 0.01,
#     'top_k_list': [50, 100, 200, 500],  # Added 500 for real data
#     'lsh_bits': 32,
#     'lsh_tables': 4,
#     'train_ratio': 0.2,
# }

# CONFIG = {
#     'embed_dim': 384,
#     'top_k_list': [50, 100, 200, 500],
#     'lsh_bits': 64,          # Increased from 32
#     'lsh_tables': 8,         # Increased from 4
#     'train_ratio': 0.2,
# }

# CONFIG = {
#     'embed_dim': 832,                # 768 (mpnet) + 64 (structural)
#     'top_k_list': [50, 100, 200, 500],
#     'lsh_bits': 64,                  # LSH signature length
#     'lsh_tables': 8,                 # number of LSH tables
#     'train_ratio': 0.2,              # 20% seeds for training (matches EIEA)
#     'use_structure': True,           # enable structural features
#     'seed': 42,                      # reproducibility
# }

# config.py
CONFIG = {
    'embed_dim': 448,  # 384 (BERT) + 64 (structural)
    'top_k_list': [50, 100, 200, 500],
    'lsh_bits': 64,
    'lsh_tables': 8,
    'train_ratio': 0.2,
    'use_structure': True,  # Enable structural features
}

