# This script will handle the math of combining weights based on the sample_count (n) you found in the metadata.json.
import numpy as np

def federated_average(peer_weights, peer_sample_counts):
    """
    Implements the FedAvg formula: w_global = sum((n_k / n_total) * w_k)
    :param peer_weights: List of weight lists [node1_weights, node2_weights, ...]
    :param peer_sample_counts: List of integers [n1, n2, ...]
    """
    total_samples = sum(peer_sample_counts)

    # Initialize global weights with zeros in the same shape as the first peer's weights
    global_weights = [np.zeros_like(w) for w in peer_weights[0]]

    for i in range(len(peer_weights)):
        # Calculate the weight factor (n_k / n_total)
        weight_factor = peer_sample_counts[i] / total_samples

        for j in range(len(peer_weights[i])):
            # Add the weighted contribution of this node to the global model
            global_weights[j] += peer_weights[i][j] * weight_factor

    return global_weights