import os
import sys
import time
import json
import grpc
import threading
import numpy as np

# ---------------------------------------------------------------------------
# Memory config — must happen before TensorFlow is imported.
# ---------------------------------------------------------------------------
os.environ.setdefault("TF_CPU_ALLOCATOR_USE_BFC", "1")
os.environ.setdefault("TF_BFC_ALLOCATOR_LIMIT_MB", "1000")   # Balanced memory pool cap
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import tensorflow as tf
from concurrent import futures

gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError:
        pass

import dfl_service_pb2
import dfl_service_pb2_grpc
from model import build_model, serialize_weights, deserialize_weights, get_weight_count

# --- Node configuration ---
NODE_ID         = int(os.environ.get("NODE_ID", 1))
NEIGHBORS       = [n.strip() for n in os.environ.get("NEIGHBORS", "").split(",") if n.strip()]
DATA_DIR        = os.environ.get("DATA_DIR", "/app/data")
LOCAL_EPOCHS    = int(os.environ.get("LOCAL_EPOCHS", 5))
MAX_ROUNDS      = int(os.environ.get("MAX_ROUNDS", 40))
NUM_CLASSES     = int(os.environ.get("NUM_CLASSES", 26))
FINE_TUNE_ROUND = int(os.environ.get("FINE_TUNE_ROUND", 0))

CHECKPOINT_DIR  = f"/app/checkpoints/node{NODE_ID}"
EXPORT_DIR      = "/app/exported_models"
GLOBAL_VAL_DIR  = os.environ.get("GLOBAL_VAL_DIR", "")
AUTOTUNE        = tf.data.AUTOTUNE
IMG_SIZE        = (128, 128)
BATCH_SIZE      = int(os.environ.get("BATCH_SIZE", 8))


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------
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
            print(f"[Node {NODE_ID}] Resumed from round {round_num}.", flush=True)
            return round_num, round_history
        except Exception as e:
            print(f"[Node {NODE_ID}] WARNING: checkpoint load failed: {e}. Starting fresh.", flush=True)
    return 0, []


def export_model(model, round_history, num_classes):
    os.makedirs(EXPORT_DIR, exist_ok=True)
    model_path = os.path.join(EXPORT_DIR, f"node{NODE_ID}_final.keras")
    model.save(model_path)
    print(f"[Node {NODE_ID}] Model exported → {model_path}", flush=True)

    label_map = {i: chr(ord('A') + i) for i in range(num_classes)}
    with open(os.path.join(EXPORT_DIR, f"node{NODE_ID}_class_map.json"), "w") as f:
        json.dump(label_map, f, indent=4)

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
        print(f"[Node {NODE_ID}] Peak val acc: {best[2]*100:.2f}% at round {best[0]}", flush=True)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def count_samples_in_dir(directory):
    total = 0
    for root, _, files in os.walk(directory):
        total += sum(1 for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')))
    return total


# Target-specific data augmentation (Sign-to-speech: horizontal flip is DISABLED)
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomRotation(0.06),
    tf.keras.layers.RandomTranslation(0.05, 0.05),
    tf.keras.layers.RandomZoom(0.05),
    tf.keras.layers.RandomBrightness(0.05),
], name='augmentation')


def make_dataset(directory, shuffle, augment=False, cache=False):
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"Dataset directory not found: {directory}")

    ds = tf.keras.utils.image_dataset_from_directory(
        directory,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        shuffle=shuffle,
    )

    # OPTIMIZATION: Switched num_parallel_calls to AUTOTUNE to clear disk I/O bottlenecks
    ds = ds.map(lambda x, y: (tf.cast(x, tf.float32), y), num_parallel_calls=AUTOTUNE)

    if cache:
        ds = ds.cache()
    if shuffle:
        ds = ds.shuffle(buffer_size=1500, reshuffle_each_iteration=True)
    if augment:
        ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE)

    return ds.prefetch(buffer_size=AUTOTUNE)


def load_training_data():
    train_dir = os.path.join(DATA_DIR, "train")
    val_dir   = os.path.join(DATA_DIR, "val")

    if not os.path.isdir(train_dir) or not os.path.isdir(val_dir):
        raise FileNotFoundError(f"[Node {NODE_ID}] Train/Val directory components missing.")

    train_samples = count_samples_in_dir(train_dir)
    val_samples   = count_samples_in_dir(val_dir)

    train_ds = make_dataset(train_dir, shuffle=True,  augment=True,  cache=False)
    val_ds   = make_dataset(val_dir,   shuffle=False, augment=False, cache=True)

    train_ds.samples = train_samples
    val_ds.samples   = val_samples

    print(f"[Node {NODE_ID}] Dataset shards bound: {train_samples} train, {val_samples} val.", flush=True)
    return train_ds, val_ds


def load_global_val():
    if not GLOBAL_VAL_DIR or not os.path.isdir(GLOBAL_VAL_DIR):
        return None
    try:
        ds = make_dataset(GLOBAL_VAL_DIR, shuffle=False, augment=False, cache=True)
        print(f"[Node {NODE_ID}] Shared verification set initialized.", flush=True)
        return ds
    except Exception as e:
        print(f"[Node {NODE_ID}] Verification load skipped: {e}", flush=True)
        return None


# ---------------------------------------------------------------------------
# Local training & Architecture adjustments
# ---------------------------------------------------------------------------
def train_local(model, train_data, val_data, global_val_data, epochs=1):
    if train_data is None:
        return None, None

    history = model.fit(
        train_data,
        epochs=epochs,
        verbose=0,
        validation_data=val_data,
    )
    local_acc  = history.history.get("val_accuracy", [0])[-1]
    local_loss = history.history.get("val_loss",     [0])[-1]
    print(f"[Node {NODE_ID}] metrics -> local_loss={local_loss:.4f} local_acc={local_acc*100:.2f}%", flush=True)

    if global_val_data is not None:
        g_loss, g_acc = model.evaluate(global_val_data, verbose=0)
        print(f"[Node {NODE_ID}] metrics -> global_loss={g_loss:.4f} global_acc={g_acc*100:.2f}%", flush=True)

    return local_loss, local_acc


def apply_fine_tuning(model, num_unfreeze=20):
    from tensorflow.keras import layers as klayers
    base_model = next(
        (l for l in model.layers if hasattr(l, 'layers') and 'mobilenetv2' in l.name.lower()),
        None
    )
    if base_model is None:
        print(f"[Node {NODE_ID}] Sub-model match missing, fine-tune skipped.", flush=True)
        return model

    base_model.trainable = True
    for layer in base_model.layers[:-num_unfreeze]:
        layer.trainable = False
    for layer in base_model.layers:
        if isinstance(layer, klayers.BatchNormalization):
            layer.trainable = False

    print(f"[Node {NODE_ID}] Backbone unfrozen for deep configuration tuning.", flush=True)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


# ---------------------------------------------------------------------------
# FedAvg Execution
# ---------------------------------------------------------------------------
def fedavg(local_flat, local_n, peer_weights):
    all_weights = [(local_flat, local_n)] + peer_weights
    total_n = sum(n for _, n in all_weights)
    if total_n == 0:
        return local_flat
    return (sum(w * n for w, n in all_weights) / total_n).astype(np.float32)


def apply_aggregated_weights(model, aggregated_flat):
    shapes = [w.shape for w in model.get_weights()]
    new_weights, offset = [], 0
    for shape in shapes:
        size = int(np.prod(shape))
        new_weights.append(aggregated_flat[offset:offset + size].reshape(shape))
        offset += size
    model.set_weights(new_weights)


# --- Bootstrap Node Configurations ---
local_n = max(count_samples_in_dir(os.path.join(DATA_DIR, "train")), 1)
train_data, val_data = load_training_data()
global_val_data = load_global_val()

model = build_model(num_classes=NUM_CLASSES)
round_num, round_history = load_checkpoint(model)
fine_tuning_applied = False

if FINE_TUNE_ROUND > 0 and round_num >= FINE_TUNE_ROUND:
    apply_fine_tuning(model, num_unfreeze=20)
    fine_tuning_applied = True

received_buffer = []
buffer_lock = threading.Lock()

grpc_options = [
    ("grpc.max_send_message_length",    32 * 1024 * 1024),
    ("grpc.max_receive_message_length", 32 * 1024 * 1024),
]


# ---------------------------------------------------------------------------
# gRPC Communication Service Handler
# ---------------------------------------------------------------------------
class DFLServicer(dfl_service_pb2_grpc.DFLServiceServicer):
    def GossipWeights(self, request, context):
        flat = np.frombuffer(request.model_data, dtype=np.float32).copy()
        peer_sample_count = request.sample_count if request.sample_count > 0 else 1
        with buffer_lock:
            received_buffer.append((flat, peer_sample_count))
        print(f"[Node {NODE_ID}] Inbound weights from Node {request.node_id}. Queue size: {len(received_buffer)}", flush=True)
        return dfl_service_pb2.WeightResponse(success=True)


def serve():
    # Downscaled threads slightly to minimize Docker background context-switching overhead
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2), options=grpc_options)
    dfl_service_pb2_grpc.add_DFLServiceServicer_to_server(DFLServicer(), server)
    server.add_insecure_port("[::]:50051")
    server.start()
    print(f"[Node {NODE_ID}] gRPC server listening. Activating high-speed decentralized loop...", flush=True)

    global round_num, round_history, fine_tuning_applied

    while round_num < MAX_ROUNDS:
        round_num += 1
        print(f"\n[Node {NODE_ID}] ===== FL ROUND {round_num}/{MAX_ROUNDS} =====", flush=True)

        if FINE_TUNE_ROUND > 0 and round_num >= FINE_TUNE_ROUND and not fine_tuning_applied:
            apply_fine_tuning(model, num_unfreeze=20)
            fine_tuning_applied = True

        # --- STEP 1: Process and clear incoming queue immediately ---
        with buffer_lock:
            snapshot = received_buffer.copy()
            received_buffer.clear()

        if snapshot:
            local_flat = np.concatenate([w.flatten() for w in model.get_weights()]).astype(np.float32)
            compatible = [(f, n) for f, n in snapshot if len(f) == len(local_flat)]

            if compatible:
                aggregated = fedavg(local_flat, local_n, compatible)
                apply_aggregated_weights(model, aggregated)
                print(f"[Node {NODE_ID}] Aggregated global states with {len(compatible)} active peers.", flush=True)
        else:
            print(f"[Node {NODE_ID}] Inbound queue empty — using current local parameters.", flush=True)

        # --- STEP 2: Train locally on the freshly combined parameters ---
        loss, acc = train_local(model, train_data, val_data, global_val_data, epochs=LOCAL_EPOCHS)
        if acc is not None:
            round_history.append((round_num, loss, acc))

        # --- STEP 3: Checkpoint updates ---
        save_checkpoint(model, round_num, round_history)

        # --- STEP 4: Asynchronously dispatch updates across the network mesh ---
        if NEIGHBORS:
            payload = serialize_weights(model)
            for target in NEIGHBORS:
                try:
                    with grpc.insecure_channel(f"{target}:50051", options=grpc_options) as channel:
                        stub = dfl_service_pb2_grpc.DFLServiceStub(channel)
                        stub.GossipWeights(
                            dfl_service_pb2.WeightRequest(
                                node_id=NODE_ID,
                                model_data=payload,
                                sample_count=local_n,
                            ),
                            timeout=8,
                        )
                except Exception:
                    pass  # Quietly bypass any lagging or starting neighbors to maintain speed

        # OPTIMIZATION: Light network clearance buffer replaces the old heavy 60-second sleep
        time.sleep(5)

    print(f"\n[Node {NODE_ID}] ===== WORKLOAD RUN COMPLETE =====", flush=True)
    export_model(model, round_history, NUM_CLASSES)
    server.stop(0)


if __name__ == "__main__":
    serve()