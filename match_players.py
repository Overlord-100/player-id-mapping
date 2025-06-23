import pickle
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.pairwise import cosine_similarity

# Load feature dictionaries
with open("broadcast_features.pkl", "rb") as f:
    broadcast_features = pickle.load(f)

with open("tacticam_features.pkl", "rb") as f:
    tacticam_features = pickle.load(f)


broadcast_ids = list(broadcast_features.keys())
tacticam_ids = list(tacticam_features.keys())

print("📦 Loaded features:")
print(f"  Broadcast players: {len(broadcast_features)}")
print(f"  Tacticam players: {len(tacticam_features)}")

# Convert to matrices
broadcast_matrix = np.array([broadcast_features[pid] for pid in broadcast_ids])
tacticam_matrix = np.array([tacticam_features[pid] for pid in tacticam_ids])

# Compute cosine distance matrix (1 - similarity)
cos_sim = cosine_similarity(tacticam_matrix, broadcast_matrix)
cos_dist = 1 - cos_sim

# Use Hungarian algorithm to find best matches
row_ind, col_ind = linear_sum_assignment(cos_dist)

# Create mapping dictionary: tacticam_id -> broadcast_id
mapping = {}
for i, j in zip(row_ind, col_ind):
    tacticam_id = tacticam_ids[i]
    broadcast_id = broadcast_ids[j]
    mapping[tacticam_id] = broadcast_id

# Output result
print("🎯 Matched Player IDs (tacticam → broadcast):")
for tid, bid in mapping.items():
    print(f"Tacticam ID {tid} → Broadcast ID {bid}")
