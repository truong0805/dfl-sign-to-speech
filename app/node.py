import os
import sys
import time
import json
import grpc
import threading
import numpy as np

# ---------------------------------------------------------------------------
# Memory & thread config — MUST happen before TensorFlow is imported.
# ---------------------------------------------------------------------------
os.environ.setdefault("TF_CPU_ALLOCATOR_USE_BFC", "1")
os.environ.setdefault("TF_BFC_ALLOCATOR_LIMIT_MB", "400")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import tensorflow as tf
from concurrent import futures

tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

import dfl_service_pb2
import dfl_service_pb2_grpc
from model import build_model, serialize_weights, get_weight_count

# ---------------------------------------------------------------------------
# Node configuration
# ---------------------------------------------------------------------------
NODE_ID         = int(os.environ.get("NODE_ID", 1))
NEIGHBORS       = [n.strip() for n in os.environ.get("NEIGHBORS", "").split(",") if n.strip()]
DATA_DIR        = os.environ.get("DATA_DIR", "/app/data")
LOCAL_EPOCHS    = int(os.environ.get("LOCAL_EPOCHS", 5))
MAX_ROUNDS      = int(os.environ.get("MAX_ROUNDS", 30))
NUM_CLASSES     = int(os.environ.get("NUM_CLASSES", 26))
FINE_TUNE_ROUND = int(os.environ.get("FINE_TUNE_ROUND", 0))
GOSSIP_INTERVAL = int(os.environ.get("GOSSIP_INTERVAL", 30))

CHECKPOINT_DIR = f"/app/checkpoints/node{NODE_ID}"
EXPORT_DIR     = "/app/exported_models"
GLOBAL_VAL_DIR = os.environ.get("GLOBAL_VAL_DIR", "")
IMG_SIZE       = (160, 160)    # Bumped from 128×128 — resolves finger details for similar signs
BATCH_SIZE     = int(os.environ.get("BATCH_SIZE", 8))

# Maximum number of peer weight sets to keep in the buffer at once.
# Each entry is ~9 MB (2.3M float32 params). Capping at 10 = ~90 MB max
# for the buffer, preventing OOM if rounds overlap or a node falls behind.
MAX_BUFFER_SIZE = 10


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
            print(f"[Node {NODE_ID}] Checkpoint load failed: {e}. Starting fresh.", flush=True)
    return 0, []


def export_model(model, round_history, num_classes):
    os.makedirs(EXPORT_DIR, exist_ok=True)
    model.save(os.path.join(EXPORT_DIR, f"node{NODE_ID}_final.keras"))
    print(f"[Node {NODE_ID}] Model exported.", flush=True)

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
        print(f"[Node {NODE_ID}] Peak val_acc={best[2]*100:.2f}% at round {best[0]}", flush=True)


# ---------------------------------------------------------------------------
# Data pipeline
# ---------------------------------------------------------------------------
def count_samples_in_dir(directory):
    total = 0
    for root, _, files in os.walk(directory):
        total += sum(1 for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')))
    return total


# Augmentation targeting the three real gaps between studio training data
# and live webcam input:
#
#   - Zoom (-0.35, +0.10)  : zoom-out up to 35%, zoom-in up to 10%.
#                             Simulates the hand being further from the webcam.
#                             Training data is close-up studio shots; live webcam
#                             has the hand as a smaller fraction of the frame.
#                             fill_mode='constant', fill_value=127: exposed border
#                             filled with neutral grey, matching training backgrounds.
#   - Brightness ±35%      : webcam auto-exposure creates wider brightness swings
#                             than controlled studio lighting.
#   - Contrast ±35%        : flat indoor lighting vs outdoor/studio light.
#
#   Deliberately removed:
#   - RandomRotation       : camera is fixed; rotation doesn't close any real
#                             domain gap and risks confusing orientation-sensitive
#                             signs (e.g. D vs G, U vs H).
#   - RandomTranslation    : MediaPipe already centres the hand in the crop,
#                             so translation variation is handled by detection,
#                             not by the classifier.
#   - Horizontal flip      : ASL is NOT mirror-symmetric (e.g. J, Z, G, H).
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomZoom(
        (-0.35, 0.10),
        fill_mode='constant',
        fill_value=127,           # neutral grey — matches plain-wall training backgrounds
    ),
    tf.keras.layers.RandomBrightness(0.35),
    tf.keras.layers.RandomContrast(0.35),
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
    ds = ds.map(lambda x, y: (tf.cast(x, tf.float32), y), num_parallel_calls=1)

    if cache:
        ds = ds.cache()
    if shuffle:
        ds = ds.shuffle(buffer_size=500, reshuffle_each_iteration=True)
    if augment:
        ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y),
                    num_parallel_calls=1)

    return ds.prefetch(buffer_size=1)


def load_training_data():
    train_dir = os.path.join(DATA_DIR, "train")
    val_dir   = os.path.join(DATA_DIR, "val")

    if not os.path.isdir(train_dir):
        raise FileNotFoundError(f"[Node {NODE_ID}] train/ missing: {train_dir}")
    if not os.path.isdir(val_dir):
        raise FileNotFoundError(f"[Node {NODE_ID}] val/ missing: {val_dir}")

    train_samples = count_samples_in_dir(train_dir)
    val_samples   = count_samples_in_dir(val_dir)

    train_ds = make_dataset(train_dir, shuffle=True,  augment=True,  cache=False)
    val_ds   = make_dataset(val_dir,   shuffle=False, augment=False, cache=False)

    train_ds.samples = train_samples
    val_ds.samples   = val_samples

    print(f"[Node {NODE_ID}] Data: {train_samples} train, {val_samples} val "
          f"| img={IMG_SIZE} batch={BATCH_SIZE}", flush=True)
    return train_ds, val_ds


def load_global_val():
    if not GLOBAL_VAL_DIR or not os.path.isdir(GLOBAL_VAL_DIR):
        return None
    try:
        ds = make_dataset(GLOBAL_VAL_DIR, shuffle=False, augment=False, cache=False)
        print(f"[Node {NODE_ID}] Global val set loaded.", flush=True)
        return ds
    except Exception as e:
        print(f"[Node {NODE_ID}] Global val skipped: {e}", flush=True)
        return None


# ---------------------------------------------------------------------------
# Local training
# ---------------------------------------------------------------------------
def train_local(model, train_data, val_data, global_val_data, epochs=LOCAL_EPOCHS):
    """
    Train for up to `epochs` local epochs with early stopping (patience=3).

    Early stopping prevents each node from over-specialising on its own shard
    when val_accuracy stops improving — common after FedAvg merges stabilise
    the weights. restore_best_weights=True ensures the checkpoint always holds
    the best seen weights from this round, not the last epoch's weights.

    Patience increased from 2 → 3: with batch_size=8 and FedAvg merges
    disrupting weights each round, patience=2 was too aggressive and would
    stop training before the model could recover from the merge.
    """
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=3,
        restore_best_weights=True,
        verbose=0,
    )

    history = model.fit(
        train_data,
        epochs=epochs,
        validation_data=val_data,
        callbacks=[early_stop],
        verbose=0,
    )

    # Last epoch's train metrics
    train_loss = history.history['loss'][-1]
    train_acc  = history.history['accuracy'][-1]
    epochs_run = len(history.history['loss'])

    # Evaluate on local val set (uses restore_best_weights result)
    local_loss, local_acc = model.evaluate(val_data, verbose=0)

    print(f"[Node {NODE_ID}] "
          f"train loss={train_loss:.4f} acc={train_acc*100:.2f}%  |  "
          f"local  val_loss={local_loss:.4f}  val_acc={local_acc*100:.2f}%  "
          f"[{epochs_run}/{epochs} epochs]",
          flush=True)

    # Optional global val (requires /app/global_val to be populated)
    if global_val_data is not None:
        g_loss, g_acc = model.evaluate(global_val_data, verbose=0)
        print(f"[Node {NODE_ID}] global val_loss={g_loss:.4f}  val_acc={g_acc*100:.2f}%",
              flush=True)

    return local_loss, local_acc


# ---------------------------------------------------------------------------
# Fine-tuning
# ---------------------------------------------------------------------------
def apply_fine_tuning(model, num_unfreeze=30):
    from tensorflow.keras import layers as klayers
    base_model = next(
        (l for l in model.layers if hasattr(l, 'layers') and 'mobilenetv2' in l.name.lower()),
        None
    )
    if base_model is None:
        print(f"[Node {NODE_ID}] MobileNetV2 sub-model not found — fine-tune skipped.", flush=True)
        return model

    base_model.trainable = True
    for layer in base_model.layers[:-num_unfreeze]:
        layer.trainable = False
    for layer in base_model.layers:
        if isinstance(layer, klayers.BatchNormalization):
            layer.trainable = False

    trainable = sum(1 for l in base_model.layers if l.trainable)
    print(f"[Node {NODE_ID}] Fine-tuning: unfroze top {trainable} MobileNetV2 layers.", flush=True)

    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.optimizers.schedules import CosineDecayRestarts

    ft_schedule = CosineDecayRestarts(
        initial_learning_rate=1e-5,
        first_decay_steps=2500,
        t_mul=1.0,
        alpha=1e-7,
    )
    model.compile(
        optimizer=Adam(ft_schedule),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=['accuracy'],
    )
    return model


# ---------------------------------------------------------------------------
# FedAvg
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


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------
local_n        = max(count_samples_in_dir(os.path.join(DATA_DIR, "train")), 1)
train_data, val_data = load_training_data()
global_val_data      = load_global_val()

model = build_model(num_classes=NUM_CLASSES)
print(f"[Node {NODE_ID}] Model built. Total weights: {get_weight_count(model):,}", flush=True)

round_num, round_history = load_checkpoint(model)
fine_tuning_applied = False
if FINE_TUNE_ROUND > 0 and round_num >= FINE_TUNE_ROUND:
    apply_fine_tuning(model, num_unfreeze=30)
    fine_tuning_applied = True

received_buffer = []
buffer_lock     = threading.Lock()

grpc_options = [
    ("grpc.max_send_message_length",    32 * 1024 * 1024),
    ("grpc.max_receive_message_length", 32 * 1024 * 1024),
]


# ---------------------------------------------------------------------------
# gRPC service
# ---------------------------------------------------------------------------
class DFLServicer(dfl_service_pb2_grpc.DFLServiceServicer):
    def GossipWeights(self, request, context):
        flat   = np.frombuffer(request.model_data, dtype=np.float32).copy()
        peer_n = request.sample_count if request.sample_count > 0 else 1
        with buffer_lock:
            if len(received_buffer) < MAX_BUFFER_SIZE:
                received_buffer.append((flat, peer_n))
                buf_size = len(received_buffer)
            else:
                # Drop oldest entry to make room (FIFO eviction)
                received_buffer.pop(0)
                received_buffer.append((flat, peer_n))
                buf_size = len(received_buffer)
                print(f"[Node {NODE_ID}] Buffer full — evicted oldest entry.", flush=True)
        print(f"[Node {NODE_ID}] Weights received from Node {request.node_id}. "
              f"Buffer: {buf_size}", flush=True)
        return dfl_service_pb2.WeightResponse(success=True)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2), options=grpc_options)
    dfl_service_pb2_grpc.add_DFLServiceServicer_to_server(DFLServicer(), server)
    server.add_insecure_port("[::]:50051")
    server.start()
    print(f"[Node {NODE_ID}] gRPC server online. Neighbors: {NEIGHBORS}", flush=True)

    global round_num, round_history, fine_tuning_applied

    while round_num < MAX_ROUNDS:
        time.sleep(GOSSIP_INTERVAL)

        round_num += 1
        print(f"\n[Node {NODE_ID}] ===== FL Round {round_num}/{MAX_ROUNDS} =====", flush=True)

        # Fine-tune transition
        if FINE_TUNE_ROUND > 0 and round_num >= FINE_TUNE_ROUND and not fine_tuning_applied:
            apply_fine_tuning(model, num_unfreeze=30)
            fine_tuning_applied = True

        # Step 1: Aggregate received peer weights (FedAvg), then train
        with buffer_lock:
            snapshot = received_buffer.copy()
            received_buffer.clear()

        if snapshot:
            local_flat = np.concatenate(
                [w.flatten() for w in model.get_weights()]
            ).astype(np.float32)
            compatible = [(f, n) for f, n in snapshot if len(f) == len(local_flat)]
            if len(snapshot) > len(compatible):
                print(f"[Node {NODE_ID}] Skipped {len(snapshot)-len(compatible)} "
                      f"incompatible weight(s).", flush=True)
            if compatible:
                aggregated = fedavg(local_flat, local_n, compatible)
                apply_aggregated_weights(model, aggregated)
                print(f"[Node {NODE_ID}] FedAvg with {len(compatible)} peer(s).", flush=True)
        else:
            print(f"[Node {NODE_ID}] No peer weights — training from local model.", flush=True)

        # Step 2: Local training (with early stopping)
        loss, acc = train_local(model, train_data, val_data, global_val_data, epochs=LOCAL_EPOCHS)
        if acc is not None:
            round_history.append((round_num, loss, acc))

        # Step 3: Checkpoint
        save_checkpoint(model, round_num, round_history)

        # Step 4: Gossip to all neighbors
        if NEIGHBORS:
            payload = serialize_weights(model)
            for target in NEIGHBORS:
                try:
                    with grpc.insecure_channel(f"{target}:50051", options=grpc_options) as ch:
                        stub = dfl_service_pb2_grpc.DFLServiceStub(ch)
                        stub.GossipWeights(
                            dfl_service_pb2.WeightRequest(
                                node_id=NODE_ID,
                                model_data=payload,
                                sample_count=local_n,
                            ),
                            timeout=15,
                        )
                    print(f"[Node {NODE_ID}] → gossiped to {target}.", flush=True)
                except Exception as e:
                    print(f"[Node {NODE_ID}] Gossip failed → {target}: {e}", flush=True)

        # Step 5: Clean up memory
        import gc
        gc.collect()

    # Training complete
    print(f"\n[Node {NODE_ID}] ===== TRAINING COMPLETE ({MAX_ROUNDS} rounds) =====", flush=True)
    if round_history:
        best = max(round_history, key=lambda x: x[2])
        for r, l, a in round_history:
            tag = " ← PEAK" if r == best[0] else ""
            print(f"[Node {NODE_ID}]  Round {r:3d} | loss={l:.4f} | acc={a*100:.2f}%{tag}",
                  flush=True)
    export_model(model, round_history, NUM_CLASSES)
    server.stop(0)


if __name__ == "__main__":
    serve()