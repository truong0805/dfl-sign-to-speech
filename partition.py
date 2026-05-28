import os
import shutil
import json
import random

# Classes to exclude — checked case-insensitively so it works whether
# your dataset folders are named 'del', 'Del', or 'DEL'
EXCLUDE_CLASSES = {'space', 'nothing', 'del'}

def create_shards(source_dir, output_root, num_nodes=5, shards_per_class=5, val_split=0.2):
    """
    Partition a resized dataset into per-node train/val shards for DFL.

    FIX 1: shards_per_node is now computed automatically from the actual
    number of classes found, instead of being hardcoded to 26. If your new
    dataset has a different number of classes, it still works correctly.

    FIX 2: EXCLUDE_CLASSES comparison is now case-insensitive, so folder
    names like 'Del', 'Space', 'Nothing' are also excluded correctly.

    FIX 3: Added warning if total shards don't divide evenly across nodes,
    so you know immediately if any images are being silently dropped.

    metadata.json is written automatically for each node — no extra steps needed.
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

    # Compute shards_per_node automatically:
    # Total shards = num_classes * shards_per_class
    # Each node gets total_shards / num_nodes
    total_shards    = num_classes * shards_per_class
    shards_per_node = total_shards // num_nodes

    leftover = total_shards % num_nodes
    if leftover != 0:
        print(f"WARNING: {total_shards} total shards cannot divide evenly into {num_nodes} nodes.")
        print(f"  {leftover} shard(s) will be unused. Consider adjusting shards_per_class.")

    print(f"Total shards: {total_shards} | Shards per node: {shards_per_node}")

    # Create node folder structure
    node_paths = [os.path.join(output_root, f"node{i+1}") for i in range(num_nodes)]
    for path in node_paths:
        for cls in classes:
            os.makedirs(os.path.join(path, "train", cls), exist_ok=True)
            os.makedirs(os.path.join(path, "val",   cls), exist_ok=True)

    # 2. Build shard pool
    all_shards = []
    for cls in classes:
        cls_path = os.path.join(source_dir, cls)
        images = [
            os.path.join(cls, img)
            for img in os.listdir(cls_path)
            if os.path.isfile(os.path.join(cls_path, img))
        ]
        random.shuffle(images)
        size = len(images) // shards_per_class
        if size == 0:
            print(f"WARNING: class '{cls}' has fewer images than shards_per_class={shards_per_class}. Skipping.")
            continue
        for i in range(shards_per_class):
            shard_data = images[i * size : (i + 1) * size]
            all_shards.append({"label": cls, "data": shard_data})
    random.shuffle(all_shards)

    # 3. Assign shards to nodes
    node_metadata = {
        i: {"node_id": i + 1, "sample_count": 0, "val_count": 0, "labels": {}}
        for i in range(num_nodes)
    }

    for i in range(num_nodes):
        assigned = all_shards[i * shards_per_node : (i + 1) * shards_per_node]
        if not assigned:
            print(f"WARNING: Node {i+1} received no shards — check your class/shard counts.")
            continue

        for shard in assigned:
            label  = shard["label"]
            images = shard["data"]

            random.shuffle(images)
            split_idx    = int(len(images) * (1 - val_split))
            train_images = images[:split_idx]
            val_images   = images[split_idx:]

            # Copy train files
            train_dir = os.path.join(node_paths[i], "train", label)
            os.makedirs(train_dir, exist_ok=True)
            for img_rel_path in train_images:
                src = os.path.join(source_dir, img_rel_path)
                dst = os.path.join(train_dir, os.path.basename(img_rel_path))
                shutil.copy(src, dst)

            # Copy val files
            val_dir = os.path.join(node_paths[i], "val", label)
            os.makedirs(val_dir, exist_ok=True)
            for img_rel_path in val_images:
                src = os.path.join(source_dir, img_rel_path)
                dst = os.path.join(val_dir, os.path.basename(img_rel_path))
                shutil.copy(src, dst)

            train_count = len(train_images)
            node_metadata[i]["sample_count"] += train_count
            node_metadata[i]["val_count"]     += len(val_images)
            node_metadata[i]["labels"][label]  = node_metadata[i]["labels"].get(label, 0) + train_count

    # 4. Save metadata.json per node
    print("\nPartition summary:")
    for i in range(num_nodes):
        meta_path = os.path.join(node_paths[i], "metadata.json")
        with open(meta_path, "w") as f:
            json.dump(node_metadata[i], f, indent=4)

        train_count = node_metadata[i]["sample_count"]
        val_count   = node_metadata[i]["val_count"]
        print(f"  Node {i+1}: {train_count} train | {val_count} val | classes: {sorted(node_metadata[i]['labels'].keys())}")


if __name__ == "__main__":
    create_shards(
        source_dir='resized_dataset',
        output_root='data_shards',
        num_nodes=5,
        shards_per_class=5,   # 26 classes × 5 shards = 130 total → 26 shards/node
        val_split=0.2,
    )
    print("\nPartitioning complete.")