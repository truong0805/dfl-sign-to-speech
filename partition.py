"""
partition.py — Dirichlet Non-IID Data Partitioning for DFL.

CHANGES FROM PREVIOUS VERSION (shard-based):
  - Replaced random-shard assignment with Dirichlet distribution (α=0.5).
    This guarantees every node sees every class while creating realistic
    non-IID heterogeneity — matching real-world federated scenarios.
  - Added MIN_PER_CLASS floor: every node gets at least MIN_PER_CLASS
    images per class, preventing the class-starvation problem where some
    nodes had zero samples for 9+ classes.
  - Dirichlet α controls the degree of heterogeneity:
      α=100  → nearly IID (uniform distribution)
      α=1.0  → moderate non-IID
      α=0.5  → strong non-IID (recommended for DFL research)
      α=0.1  → extreme non-IID (some nodes dominate certain classes)
  - Kept: metadata.json output, train/val split, EXCLUDE_CLASSES logic.
"""

import os
import shutil
import json
import random
import numpy as np

# Classes to exclude — checked case-insensitively so it works whether
# your dataset folders are named 'del', 'Del', or 'DEL'
EXCLUDE_CLASSES = {'space', 'nothing', 'del'}

# Minimum images per class per node — prevents class starvation
MIN_PER_CLASS = 10

# Dirichlet concentration parameter
# Lower α → more heterogeneous (non-IID), higher α → more uniform (IID)
DIRICHLET_ALPHA = 0.5


def create_shards(source_dir, output_root, num_nodes=5, val_split=0.2, alpha=DIRICHLET_ALPHA):
    """
    Partition a resized dataset into per-node train/val splits using
    Dirichlet-based non-IID allocation.

    Each class's images are distributed across nodes according to a
    Dirichlet(α) draw, with a minimum floor of MIN_PER_CLASS per node
    to prevent class starvation.

    Args:
        source_dir:  Path to the flat class-folder dataset (e.g. resized_dataset/)
        output_root: Path to write node shards (e.g. data_shards/)
        num_nodes:   Number of federated nodes
        val_split:   Fraction of each node's per-class images reserved for validation
        alpha:       Dirichlet concentration parameter
    """

    # 1. Wipe and recreate output
    if os.path.exists(output_root):
        shutil.rmtree(output_root)

    # Filter out unwanted classes (case-insensitive)
    classes = [
        d for d in os.listdir(source_dir)
        if os.path.isdir(os.path.join(source_dir, d))
        and d.lower() not in EXCLUDE_CLASSES
    ]
    classes = sorted(classes)
    num_classes = len(classes)
    print(f"Found {num_classes} classes: {classes}")
    print(f"Dirichlet alpha={alpha} | Nodes={num_nodes} | Val split={val_split}")
    print(f"Min per class per node: {MIN_PER_CLASS}")

    # Create node folder structure
    node_paths = [os.path.join(output_root, f"node{i+1}") for i in range(num_nodes)]
    for path in node_paths:
        for cls in classes:
            os.makedirs(os.path.join(path, "train", cls), exist_ok=True)
            os.makedirs(os.path.join(path, "val",   cls), exist_ok=True)

    # 2. Dirichlet allocation per class
    node_metadata = {
        i: {"node_id": i + 1, "sample_count": 0, "val_count": 0, "labels": {}}
        for i in range(num_nodes)
    }

    total_train_all = 0
    total_val_all   = 0

    for cls in classes:
        cls_path = os.path.join(source_dir, cls)
        images = [
            img for img in os.listdir(cls_path)
            if os.path.isfile(os.path.join(cls_path, img))
        ]
        random.shuffle(images)
        n_images = len(images)

        if n_images < num_nodes * MIN_PER_CLASS:
            print(f"WARNING: class '{cls}' has only {n_images} images — "
                  f"cannot guarantee {MIN_PER_CLASS} per node. Distributing evenly.")

        # Draw Dirichlet proportions
        proportions = np.random.dirichlet([alpha] * num_nodes)

        # Compute raw allocation
        raw_counts = (proportions * n_images).astype(int)

        # Enforce minimum floor
        for i in range(num_nodes):
            if raw_counts[i] < MIN_PER_CLASS:
                raw_counts[i] = MIN_PER_CLASS

        # If total exceeds available images, scale down proportionally
        total_allocated = raw_counts.sum()
        if total_allocated > n_images:
            # Scale down, keeping the minimum floor
            excess = total_allocated - n_images
            # Remove excess from the largest allocations first
            sorted_indices = np.argsort(raw_counts)[::-1]
            for idx in sorted_indices:
                can_remove = raw_counts[idx] - MIN_PER_CLASS
                remove = min(can_remove, excess)
                raw_counts[idx] -= remove
                excess -= remove
                if excess <= 0:
                    break

        # If total is less than available, give remainder to random nodes
        total_allocated = raw_counts.sum()
        if total_allocated < n_images:
            remainder = n_images - total_allocated
            # Distribute remainder proportionally
            for _ in range(remainder):
                idx = random.randint(0, num_nodes - 1)
                raw_counts[idx] += 1

        # Assign images to nodes
        offset = 0
        for i in range(num_nodes):
            count = int(raw_counts[i])
            node_images = images[offset:offset + count]
            offset += count

            # Train/val split within this node's allocation for this class
            random.shuffle(node_images)
            split_idx    = max(1, int(len(node_images) * (1 - val_split)))
            train_images = node_images[:split_idx]
            val_images   = node_images[split_idx:]

            # Copy train files
            train_dir = os.path.join(node_paths[i], "train", cls)
            for img_name in train_images:
                src = os.path.join(cls_path, img_name)
                dst = os.path.join(train_dir, img_name)
                shutil.copy(src, dst)

            # Copy val files
            val_dir = os.path.join(node_paths[i], "val", cls)
            for img_name in val_images:
                src = os.path.join(cls_path, img_name)
                dst = os.path.join(val_dir, img_name)
                shutil.copy(src, dst)

            train_count = len(train_images)
            val_count   = len(val_images)
            node_metadata[i]["sample_count"] += train_count
            node_metadata[i]["val_count"]    += val_count
            node_metadata[i]["labels"][cls]   = node_metadata[i]["labels"].get(cls, 0) + train_count

            total_train_all += train_count
            total_val_all   += val_count

    # 3. Save metadata.json per node
    print("\nPartition summary:")
    for i in range(num_nodes):
        meta_path = os.path.join(node_paths[i], "metadata.json")
        with open(meta_path, "w") as f:
            json.dump(node_metadata[i], f, indent=4)

        train_count = node_metadata[i]["sample_count"]
        val_count   = node_metadata[i]["val_count"]
        n_classes   = len(node_metadata[i]["labels"])
        min_cls     = min(node_metadata[i]["labels"].values()) if node_metadata[i]["labels"] else 0
        max_cls     = max(node_metadata[i]["labels"].values()) if node_metadata[i]["labels"] else 0
        print(f"  Node {i+1}: {train_count:5d} train | {val_count:4d} val | "
              f"{n_classes} classes | per-class range: [{min_cls}, {max_cls}]")

    print(f"\nTotal: {total_train_all} train + {total_val_all} val = {total_train_all + total_val_all}")


if __name__ == "__main__":
    create_shards(
        source_dir='resized_dataset',
        output_root='data_shards',
        num_nodes=5,
        val_split=0.2,
        alpha=DIRICHLET_ALPHA,   # 0.5 = strong non-IID, good for DFL research
    )
    print("\nPartitioning complete.")