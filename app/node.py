import os
import sys
import time
import json
import grpc
import threading
import numpy as np
import tensorflow as tf
from concurrent import futures
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"GPU memory growth enabled for {len(gpus)} GPU(s).", flush=True)
import dfl_service_pb2
import dfl_service_pb2_grpc
from model import build_model, serialize_weights, deserialize_weights, get_weight_count

# --- Node configuration ---
NODE_ID         = int(os.environ.get("NODE_ID", 1))
NEIGHBORS       = [n.strip() for n in os.environ.get("NEIGHBORS", "").split(",") if n.strip()]
DATA_DIR        = os.environ.get("DATA_DIR", "/app/data")
GOSSIP_INTERVAL = int(os.environ.get("GOSSIP_INTERVAL", 30))
LOCAL_EPOCHS    = int(os.environ.get("LOCAL_EPOCHS", 1))
MAX_ROUNDS      = int(os.environ.get("MAX_ROUNDS", 20))
NUM_CLASSES     = int(os.environ.get("NUM_CLASSES", 26))
CHECKPOINT_DIR  = f"/app/checkpoints/node{NODE_ID}"
EXPORT_DIR      = "/app/exported_models"
AUTOTUNE        = tf.data.AUTOTUNE
IMG_SIZE        = (128, 128)    
BATCH_SIZE      = 8  # Reduced to minimize RAM usage


def save_checkpoint(model, round_num, round_history):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    np.save(os.path.join(CHECKPOINT_DIR, "weights.npy"),
            np.array(model.get_weights(), dtype=object))
    with open(os.path.join(CHECKPOINT_DIR, "checkpoint.json"), "w") as f:
        json.dump({"round_num": round_num, "round_history": round_history}, f)
    print(f"[Node {NODE_ID}] Checkpoint saved at round {round_num}.", flush=True)


def load_checkpoint(model):
    weights_path = os.path.join(CHECKPOINT_DIR, "weights.npy")
    meta_path    = os.path.join(CHECKPOINT_DIR, "checkpoint.json")
    if os.path.exists(weights_path) and os.path.exists(meta_path):
        try:
            weights = np.load(weights_path, allow_pickle=True)
            model.set_weights(list(weights))
            with open(meta_path) as f:
                meta = json.load(f)
            round_num     = meta.get("round_num", 0)
            round_history = [tuple(r) for r in meta.get("round_history", [])]
            print(f"[Node {NODE_ID}] Resumed from checkpoint at round {round_num}.", flush=True)
            return round_num, round_history
        except Exception as e:
            print(f"[Node {NODE_ID}] WARNING: Could not load checkpoint: {e}. Starting fresh.", flush=True)
    return 0, []


def export_model(model, round_history, num_classes):
    os.makedirs(EXPORT_DIR, exist_ok=True)

    model_path = os.path.join(EXPORT_DIR, f"node{NODE_ID}_final.keras")
    model.save(model_path)
    print(f"[Node {NODE_ID}] Model exported to {model_path}", flush=True)

    # Always use canonical A-Z map — never derive from class_indices
    # (local shards may be missing some letters, causing index shifts)
    label_map = {i: chr(ord('A') + i) for i in range(num_classes)}
    map_path = os.path.join(EXPORT_DIR, f"node{NODE_ID}_class_map.json")
    with open(map_path, "w") as f:
        json.dump(label_map, f, indent=4)
    print(f"[Node {NODE_ID}] Class map exported to {map_path}", flush=True)

    if round_history:
        best = max(round_history, key=lambda x: x[2])
        summary = {
            "node_id":        NODE_ID,
            "total_rounds":   len(round_history),
            "peak_val_acc":   best[2],
            "peak_round":     best[0],
            "final_val_acc":  round_history[-1][2],
            "final_val_loss": round_history[-1][1],
        }
        with open(os.path.join(EXPORT_DIR, f"node{NODE_ID}_summary.json"), "w") as f:
            json.dump(summary, f, indent=4)
        print(f"[Node {NODE_ID}] Peak val accuracy: {best[2]*100:.2f}% at round {best[0]}", flush=True)


def load_sample_count():
    meta_path = os.path.join(DATA_DIR, "metadata.json")
    try:
        with open(meta_path) as f:
            return json.load(f).get("sample_count", 1)
    except Exception as e:
        print(f"[Node {NODE_ID}] WARNING: metadata.json error: {e}. Using 1.", flush=True)
        return 1


# ---------------------------------------------------------------------------
# In-graph augmentation (GPU-accelerated, applied only during training)
# horizontal_flip DISABLED — ASL signs are NOT mirror-symmetric.
# ---------------------------------------------------------------------------
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomRotation(0.08),
    tf.keras.layers.RandomTranslation(0.1, 0.1),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomBrightness(0.2),
    tf.keras.layers.RandomContrast(0.1),
], name='augmentation')


def load_training_data():
    """
    tf.data pipeline — faster than ImageDataGenerator.
    Images are passed as float32 [0, 255]. The model's Rescaling layer
    normalizes them to [0, 1] internally (standard ImageNet preprocessing).
    """
    train_dir = os.path.join(DATA_DIR, "train")
    val_dir   = os.path.join(DATA_DIR, "val")

    # Debug: Check if directories exist
    print(f"[Node {NODE_ID}] Looking for data in: {DATA_DIR}", flush=True)
    print(f"[Node {NODE_ID}] Train dir exists: {os.path.exists(train_dir)} ({train_dir})", flush=True)
    print(f"[Node {NODE_ID}] Val dir exists: {os.path.exists(val_dir)} ({val_dir})", flush=True)
    
    if os.path.exists(train_dir):
        train_files = os.listdir(train_dir)
        print(f"[Node {NODE_ID}] Train dir contents: {train_files[:5]}...", flush=True)

    try:
        train_ds = tf.keras.utils.image_dataset_from_directory(
            train_dir,
            image_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            label_mode="categorical",
            shuffle=True,
            class_names=[chr(ord('A') + i) for i in range(NUM_CLASSES)],
        )
        val_ds = tf.keras.utils.image_dataset_from_directory(
            val_dir,
            image_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            label_mode="categorical",
            shuffle=False,
            class_names=[chr(ord('A') + i) for i in range(NUM_CLASSES)],
        )

        num_classes = len(train_ds.class_names)

        # Count samples BEFORE consuming the dataset
        num_train_batches = len(train_ds)
        num_val_batches   = len(val_ds)
        train_samples = num_train_batches * BATCH_SIZE
        val_samples   = num_val_batches * BATCH_SIZE

        # Cast to float32 — EfficientNetB0 preprocesses internally, no /255
        def cast(x, y):
            return tf.cast(x, tf.float32), y

        def augment(x, y):
            return data_augmentation(x, training=True), y

        train_ds = (train_ds
                    .map(cast, num_parallel_calls=AUTOTUNE)
                    .map(augment, num_parallel_calls=AUTOTUNE)
                    .shuffle(100)
                    .prefetch(2))

        val_ds = (val_ds
                  .map(cast, num_parallel_calls=AUTOTUNE)
                  .prefetch(2))

        print(
            f"[Node {NODE_ID}] Loaded {train_samples} train, "
            f"{val_samples} val, {num_classes} classes.", flush=True
        )

        train_ds.samples     = train_samples
        train_ds.num_classes = num_classes
        val_ds.samples       = val_samples
        return train_ds, val_ds

    except Exception as e:
        print(f"[Node {NODE_ID}] WARNING: Could not load data: {e}.", flush=True)
        return None, None


def train_local(model, train_data, val_data, epochs=1):
    if train_data is None or train_data.samples == 0:
        print(f"[Node {NODE_ID}] No local data — skipping training.", flush=True)
        return None, None

    print(f"[Node {NODE_ID}] Training {epochs} epoch(s)...", flush=True)

    history = model.fit(
        train_data,
        epochs=epochs,
        verbose=0,
        validation_data=val_data,
    )

    acc  = history.history.get("val_accuracy", [0])[-1]
    loss = history.history.get("val_loss",     [0])[-1]
    print(f"[Node {NODE_ID}] val_loss={loss:.4f}  val_acc={acc*100:.2f}%", flush=True)
    return loss, acc


def fedavg(local_flat, local_n, peer_weights):
    all_weights = [(local_flat, local_n)] + peer_weights
    total_n = sum(n for _, n in all_weights)
    if total_n == 0:
        return local_flat
    return (sum(w * n for w, n in all_weights) / total_n).astype(np.float32)


# --- Bootstrap ---
local_n = load_sample_count()
print(f"[Node {NODE_ID}] Local sample count: {local_n}", flush=True)

train_data, val_data = load_training_data()
num_classes = int(os.environ.get("NUM_CLASSES", 26))

model = build_model(num_classes=num_classes)
print(f"[Node {NODE_ID}] Model built. Classes: {num_classes}  Weights: {get_weight_count(model)}", flush=True)

round_num, round_history = load_checkpoint(model)
if round_num > 0:
    print(f"[Node {NODE_ID}] Resuming from round {round_num}.", flush=True)

received_buffer = []
buffer_lock = threading.Lock()


class DFLServicer(dfl_service_pb2_grpc.DFLServiceServicer):
    def GossipWeights(self, request, context):
        flat = np.frombuffer(request.model_data, dtype=np.float32).copy()
        peer_sample_count = request.sample_count if request.sample_count > 0 else 1
        with buffer_lock:
            received_buffer.append((flat, peer_sample_count))
        print(
            f"[Node {NODE_ID}] Received weights from Node {request.node_id} "
            f"({len(flat)} params, {peer_sample_count} samples). Buffer: {len(received_buffer)}",
            flush=True,
        )
        return dfl_service_pb2.WeightResponse(success=True)


def serve():
    grpc_options = [
        ("grpc.max_send_message_length",    64 * 1024 * 1024),
        ("grpc.max_receive_message_length", 64 * 1024 * 1024),
    ]
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10), options=grpc_options)
    dfl_service_pb2_grpc.add_DFLServiceServicer_to_server(DFLServicer(), server)
    server.add_insecure_port("[::]:50051")
    server.start()
    print(f"[Node {NODE_ID}] gRPC server online. Neighbors: {NEIGHBORS}", flush=True)

    global round_num, round_history

    while True:
        time.sleep(GOSSIP_INTERVAL)
        round_num += 1
        print(f"[Node {NODE_ID}] ===== FL Round {round_num}{f'/{MAX_ROUNDS}' if MAX_ROUNDS else ''} =====", flush=True)

        # Step 1: Local training
        loss, acc = train_local(model, train_data, val_data, epochs=LOCAL_EPOCHS)
        if acc is not None:
            round_history.append((round_num, loss, acc))

        # Step 2: FedAvg with received peer weights
        with buffer_lock:
            snapshot = received_buffer.copy()
            received_buffer.clear()

        if snapshot:
            local_flat = np.concatenate(
                [w.ravel() for w in model.get_weights()]
            ).astype(np.float32)

            local_param_count = len(local_flat)
            compatible = [(f, n) for f, n in snapshot if len(f) == local_param_count]
            skipped = len(snapshot) - len(compatible)

            if skipped:
                print(f"[Node {NODE_ID}] Skipped {skipped} incompatible peer weights.", flush=True)

            if compatible:
                aggregated = fedavg(local_flat, local_n, compatible)
                shapes = [w.shape for w in model.get_weights()]
                new_weights, offset = [], 0
                for shape in shapes:
                    size = int(np.prod(shape))
                    new_weights.append(aggregated[offset:offset + size].reshape(shape))
                    offset += size
                model.set_weights(new_weights)
                print(f"[Node {NODE_ID}] Round {round_num}: FedAvg applied ({len(compatible)} peers).", flush=True)
            else:
                print(f"[Node {NODE_ID}] Round {round_num}: No compatible peers — keeping local model.", flush=True)
        else:
            print(f"[Node {NODE_ID}] Round {round_num}: No peer weights received.", flush=True)

        # Step 3: Save checkpoint
        save_checkpoint(model, round_num, round_history)

        # Step 4: Gossip to ALL neighbors
        if NEIGHBORS:
            payload = serialize_weights(model)
            for target in NEIGHBORS:
                try:
                    with grpc.insecure_channel(f"{target}:50051", options=grpc_options) as channel:
                        stub = dfl_service_pb2_grpc.DFLServiceStub(channel)
                        response = stub.GossipWeights(
                            dfl_service_pb2.WeightRequest(
                                node_id=NODE_ID,
                                model_data=payload,
                                sample_count=local_n,
                            )
                        )
                        if response.success:
                            print(
                                f"[Node {NODE_ID}] Round {round_num}: Gossiped to {target} ({len(payload)} bytes).",
                                flush=True,
                            )
                except Exception as e:
                    print(f"[Node {NODE_ID}] Could not reach {target}: {e}", flush=True)

        # Stop when MAX_ROUNDS reached
        if MAX_ROUNDS and round_num >= MAX_ROUNDS:
            print(f"\n[Node {NODE_ID}] ===== TRAINING COMPLETE ({MAX_ROUNDS} rounds) =====", flush=True)
            if round_history:
                best = max(round_history, key=lambda x: x[2])
                print(f"[Node {NODE_ID}] Round | Val Loss | Val Accuracy", flush=True)
                print(f"[Node {NODE_ID}] ------+----------+-------------", flush=True)
                for r, l, a in round_history:
                    marker = " <-- PEAK" if r == best[0] else ""
                    print(f"[Node {NODE_ID}]   {r:3d} | {l:.4f}   | {a*100:.2f}%{marker}", flush=True)
                    print(f"[Node {NODE_ID}] Peak val accuracy: {best[2]*100:.2f}% at round {best[0]}", flush=True)  # add this
                else:
                    print(f"[Node {NODE_ID}] No training data — no report.", flush=True)

            export_model(model, round_history, num_classes)
            server.stop(0)
            break


if __name__ == "__main__":
    serve()