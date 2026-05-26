import os
import shutil
import json
import random

# Classes to exclude
EXCLUDE_CLASSES = {'space', 'nothing', 'del'}

# Reproducibility
random.seed(42)

def create_shards(
    source_dir,
    output_root,
    num_nodes=5,
    val_split=0.2
):

    # ----------------------------
    # 1. RESET OUTPUT
    # ----------------------------
    if os.path.exists(output_root):
        shutil.rmtree(output_root)

    classes = sorted([
        d for d in os.listdir(source_dir)
        if os.path.isdir(os.path.join(source_dir, d))
        and d not in EXCLUDE_CLASSES
    ])

    print(f"\nFound {len(classes)} classes")
    print(classes)

    # ----------------------------
    # 2. CREATE NODE FOLDERS
    # ----------------------------
    node_paths = []

    for i in range(num_nodes):

        node_path = os.path.join(output_root, f"node{i+1}")

        node_paths.append(node_path)

        for split in ["train", "val"]:
            for cls in classes:
                os.makedirs(
                    os.path.join(node_path, split, cls),
                    exist_ok=True
                )

    # ----------------------------
    # 3. NODE METADATA
    # ----------------------------
    node_metadata = {
        i: {
            "node_id": i + 1,
            "sample_count": 0,
            "val_count": 0,
            "labels": {}
        }
        for i in range(num_nodes)
    }

    # ----------------------------
    # 4. SPLIT EACH CLASS
    # ----------------------------
    for cls in classes:

        cls_path = os.path.join(source_dir, cls)

        images = [
            img for img in os.listdir(cls_path)
            if img.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        random.shuffle(images)

        total = len(images)

        if total < num_nodes:
            print(f"Skipping {cls}: not enough images")
            continue

        # ----------------------------
        # GLOBAL TRAIN/VAL SPLIT
        # IMPORTANT
        # ----------------------------

        val_count = int(total * val_split)

        val_images = images[:val_count]
        train_images = images[val_count:]

        # ----------------------------
        # DISTRIBUTE TRAIN IMAGES
        # ----------------------------

        train_chunks = [[] for _ in range(num_nodes)]

        for idx, img in enumerate(train_images):
            train_chunks[idx % num_nodes].append(img)

        # ----------------------------
        # DISTRIBUTE VAL IMAGES
        # ----------------------------

        val_chunks = [[] for _ in range(num_nodes)]

        for idx, img in enumerate(val_images):
            val_chunks[idx % num_nodes].append(img)

        # ----------------------------
        # COPY FILES TO NODES
        # ----------------------------

        for node_idx in range(num_nodes):

            # TRAIN
            train_dir = os.path.join(
                node_paths[node_idx],
                "train",
                cls
            )

            for img_name in train_chunks[node_idx]:

                src = os.path.join(cls_path, img_name)

                dst = os.path.join(train_dir, img_name)

                shutil.copy(src, dst)

            # VAL
            val_dir = os.path.join(
                node_paths[node_idx],
                "val",
                cls
            )

            for img_name in val_chunks[node_idx]:

                src = os.path.join(cls_path, img_name)

                dst = os.path.join(val_dir, img_name)

                shutil.copy(src, dst)

            # ----------------------------
            # METADATA
            # ----------------------------

            train_count = len(train_chunks[node_idx])
            val_count_node = len(val_chunks[node_idx])

            node_metadata[node_idx]["sample_count"] += train_count
            node_metadata[node_idx]["val_count"] += val_count_node

            node_metadata[node_idx]["labels"][cls] = train_count

    # ----------------------------
    # 5. SAVE METADATA
    # ----------------------------

    for i in range(num_nodes):

        meta_path = os.path.join(
            node_paths[i],
            "metadata.json"
        )

        with open(meta_path, "w") as f:
            json.dump(node_metadata[i], f, indent=4)

        print(
            f"\nNode {i+1}"
            f"\nTrain: {node_metadata[i]['sample_count']}"
            f"\nVal: {node_metadata[i]['val_count']}"
        )

    print("\nPartitioning complete.")

if __name__ == "__main__":

    create_shards(
        source_dir="resized_dataset",
        output_root="data_shards",
        num_nodes=5,
        val_split=0.2
    )