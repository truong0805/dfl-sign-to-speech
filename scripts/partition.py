import os
import shutil
import json
import random
import numpy as np

def create_shards(source_dir, output_root, num_nodes=5, shards_per_class=10, shards_per_node=6):
    # 1. Setup Folders
    if os.path.exists(output_root):
        shutil.rmtree(output_root)

    classes = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]
    node_paths = [os.path.join(output_root, f"node{i+1}") for i in range(num_nodes)]

    for path in node_paths:
        os.makedirs(path, exist_ok=True)

    # 2. Build the Shard Pool
    # A shard is a list of file paths belonging to a specific class
    all_shards = []
    for cls in classes:
        cls_path = os.path.join(source_dir, cls)
        images = [os.path.join(cls, img) for img in os.listdir(cls_path)]
        random.shuffle(images)

        # Split images of this class into 10 shards
        size = len(images) // shards_per_class
        for i in range(shards_per_class):
            shard_data = images[i*size : (i+1)*size]
            all_shards.append({"label": cls, "data": shard_data})

    random.shuffle(all_shards)

    # 3. Assign Shards to Nodes
    node_metadata = {i: {"node_id": i+1, "sample_count": 0, "labels": {}} for i in range(num_nodes)}

    for i in range(num_nodes):
        # Each node gets a limited slice of the total shards
        assigned_shards = all_shards[i * shards_per_node : (i+1) * shards_per_node]

        for shard in assigned_shards:
            label = shard["label"]
            # Copy files
            target_dir = os.path.join(node_paths[i], label)
            os.makedirs(target_dir, exist_ok=True)

            for img_rel_path in shard["data"]:
                src = os.path.join(source_dir, img_rel_path)
                dst = os.path.join(target_dir, os.path.basename(img_rel_path))
                shutil.copy(src, dst)

            # Update Metadata
            count = len(shard["data"])
            node_metadata[i]["sample_count"] += count
            node_metadata[i]["labels"][label] = node_metadata[i]["labels"].get(label, 0) + count

    # 4. Save metadata.json for aggregation math later
    for i in range(num_nodes):
        meta_path = os.path.join(node_paths[i], "metadata.json")
        with open(meta_path, "w") as f:
            json.dump(node_metadata[i], f, indent=4)

if __name__ == "__main__":
    # SOURCE_DIR IS YOUR PATH TO THE DATASET
    create_shards(source_dir='dfl-sign-to-speech\resized_dataset', output_root='data_shards')
    print("Partitioning complete. 5 Non-IID nodes created with metadata.json.")