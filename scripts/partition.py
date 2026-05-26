import os
import shutil
import json
import random

# Classes to exclude
EXCLUDE_CLASSES = {'space', 'nothing', 'del'}

def create_shards(source_dir, output_root, num_nodes=5, shards_per_class=5, shards_per_node=26, val_split=0.2):
    # 1. Setup Folders
    if os.path.exists(output_root):
        shutil.rmtree(output_root)

    # Filter out unwanted classes
    classes = [
        d for d in os.listdir(source_dir)
        if os.path.isdir(os.path.join(source_dir, d)) and d not in EXCLUDE_CLASSES
    ]
    print(f"Found {len(classes)} classes: {sorted(classes)}")

    node_paths = [os.path.join(output_root, f"node{i+1}") for i in range(num_nodes)]
    for path in node_paths:
        for cls in classes:
            os.makedirs(os.path.join(path, "train", cls), exist_ok=True)
            os.makedirs(os.path.join(path, "val",   cls), exist_ok=True)

    # 2. Build the Shard Pool
    all_shards = []
    for cls in classes:
        cls_path = os.path.join(source_dir, cls)
        images = [os.path.join(cls, img) for img in os.listdir(cls_path)]
        random.shuffle(images)
        size = len(images) // shards_per_class
        for i in range(shards_per_class):
            shard_data = images[i*size : (i+1)*size]
            all_shards.append({"label": cls, "data": shard_data})
    random.shuffle(all_shards)

    # 3. Assign Shards to Nodes
    node_metadata = {i: {"node_id": i+1, "sample_count": 0, "val_count": 0, "labels": {}} for i in range(num_nodes)}
    for i in range(num_nodes):
        assigned_shards = all_shards[i * shards_per_node : (i+1) * shards_per_node]
        for shard in assigned_shards:
            label = shard["label"]
            images = shard["data"]

            # Split into train/val
            random.shuffle(images)
            split_idx = int(len(images) * (1 - val_split))
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

            # Update metadata
            train_count = len(train_images)
            node_metadata[i]["sample_count"] += train_count
            node_metadata[i]["val_count"]     += len(val_images)
            node_metadata[i]["labels"][label]  = node_metadata[i]["labels"].get(label, 0) + train_count

    # 4. Save metadata.json
    for i in range(num_nodes):
        meta_path = os.path.join(node_paths[i], "metadata.json")
        with open(meta_path, "w") as f:
            json.dump(node_metadata[i], f, indent=4)

        train_count = node_metadata[i]["sample_count"]
        val_count   = node_metadata[i]["val_count"]
        print(f"  Node {i+1}: {train_count} train, {val_count} val samples | classes: {sorted(node_metadata[i]['labels'].keys())}")

if __name__ == "__main__":
    create_shards(
        source_dir='resized_dataset',   # point this to your Kaggle dataset folder
        output_root='data_shards',
        num_nodes=5,
        shards_per_class=5,
        shards_per_node=26,         # one shard per class = all 26 letters
        val_split=0.2
    )
    print("Partitioning complete. 5 Non-IID nodes created with train/val split.")