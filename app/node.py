import os
import sys
import time
import json
import grpc
import threading
import numpy as np
from concurrent import futures
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import dfl_service_pb2
import dfl_service_pb2_grpc
from model import build_model, serialize_weights, deserialize_weights, get_weight_count

# --- Node configuration ---
NODE_ID         = int(os.environ.get("NODE_ID", 1))
NEIGHBORS       = [n.strip() for n in os.environ.get("NEIGHBORS", "").split(",") if n.strip()]
DATA_DIR        = os.environ.get("DATA_DIR", "/app/data")
GOSSIP_INTERVAL = int(os.environ.get("GOSSIP_INTERVAL", 30))
LOCAL_EPOCHS    = int(os.environ.get("LOCAL_EPOCHS", 5))
MAX_ROUNDS      = int(os.environ.get("MAX_ROUNDS", 20))
NUM_CLASSES     = int(os.environ.get("NUM_CLASSES", 26))
CHECKPOINT_DIR  = f"/app/checkpoints/node{NODE_ID}"
EXPORT_DIR      = "/app/exported_models"


def save_checkpoint(model, round_num, round_history):
    """Save model weights and training history to disk after every round."""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    np.save(os.path.join(CHECKPOINT_DIR, "weights.npy"),
            np.array(model.get_weights(), dtype=object))
    with open(os.path.join(CHECKPOINT_DIR, "checkpoint.json"), "w") as f:
        json.dump({"round_num": round_num, "round_history": round_history}, f)
    print(f"[Node {NODE_ID}] Checkpoint saved at round {round_num}.", flush=True)


def load_checkpoint(model):
    """Load weights and history from disk if a checkpoint exists."""
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


def export_model(model, round_history, train_data):
    """Save final model in Keras format with class mapping."""
    os.makedirs(EXPORT_DIR, exist_ok=True)

    model_path = os.path.join(EXPORT_DIR, f"node{NODE_ID}_final.keras")
    model.save(model_path)
    print(f"[Node {NODE_ID}] Model exported to {model_path}", flush=True)

    # Always use a canonical A-Z label map (indices 0-25).
    # Deriving from train_data.class_indices is unsafe: if a node's local
    # shard is missing any letter folder, Keras re-numbers the remaining
    # indices and the JSON map would be wrong for those positions.
    label_map = {i: chr(ord('A') + i) for i in range(NUM_CLASSES)}
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
        summary_path = os.path.join(EXPORT_DIR, f"node{NODE_ID}_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=4)
        print(f"[Node {NODE_ID}] Summary: peak={best[2]*100:.2f}% at round {best[0]}", flush=True)


def load_sample_count():
    meta_path = os.path.join(DATA_DIR, "metadata.json")
    try:
        with open(meta_path) as f:
            return json.load(f).get("sample_count", 1)
    except Exception as e:
        print(f"[Node {NODE_ID}] WARNING: Could not load metadata.json: {e}. Using 1.", flush=True)
        return 1


def load_training_data():
    """Load train/ and val/ subdirectories separately."""
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        # FIX: horizontal_flip REMOVED.
        # ASL signs are NOT mirror-symmetric. Many letters (J, Z, G, H, etc.)
        # are directional — flipping them produces a corrupted or wrong-class
        # training sample. This was a silent source of training noise.
        zoom_range=0.1,
        brightness_range=[0.8, 1.2],  # simulate lighting variation
        fill_mode='nearest',
    )
    val_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_dir = os.path.join(DATA_DIR, "train")
    val_dir   = os.path.join(DATA_DIR, "val")

    try:
        train_data = train_datagen.flow_from_directory(
            train_dir,
            target_size=(128, 128),
            batch_size=16,
            class_mode="categorical",
            shuffle=True,
        )
        val_data = val_datagen.flow_from_directory(
            val_dir,
            target_size=(128, 128),
            batch_size=16,
            class_mode="categorical",
            shuffle=False,
        )
        print(
            f"[Node {NODE_ID}] Loaded {train_data.samples} train images, "
            f"{val_data.samples} val images, {train_data.num_classes} classes.",
            flush=True,
        )
        return train_data, val_data
    except Exception as e:
        print(f"[Node {NODE_ID}] WARNING: Could not load training data: {e}. Skipping training.", flush=True)
        return None, None


def train_local(model, train_data, val_data, epochs=1):
    """Train on local shard, evaluate on local val set."""
    if train_data is None or train_data.samples == 0:
        print(f"[Node {NODE_ID}] No local data — skipping training.", flush=True)
        return None, None

    print(f"[Node {NODE_ID}] Training {epochs} epoch(s) on {train_data.samples} samples...", flush=True)

    history = model.fit(
        train_data,
        epochs=epochs,
        verbose=0,
        validation_data=val_data,
    )

    acc  = history.history.get("val_accuracy", [0])[-1]
    loss = history.history.get("val_loss",     [0])[-1]

    print(f"[Node {NODE_ID}] Local train done. val_loss={loss:.4f}  val_acc={acc * 100:.2f}%", flush=True)
    return loss, acc


def fedavg(local_flat, local_n, peer_weights):
    """Weighted FedAvg: w_new = sum(n_i * w_i) / sum(n_i)"""
    all_weights = [(local_flat, local_n)] + peer_weights
    total_n = sum(n for _, n in all_weights)
    if total_n == 0:
        return local_flat
    return (sum(w * n for w, n in all_weights) / total_n).astype(np.float32)


# --- Load data, then build model ---
local_n = load_sample_count()
print(f"[Node {NODE_ID}] Local sample count: {local_n}", flush=True)

train_data, val_data = load_training_data()
num_classes = int(os.environ.get("NUM_CLASSES", 26))

model = build_model(num_classes=num_classes)
print(f"[Node {NODE_ID}] Model built. Classes: {num_classes}  Total weights: {get_weight_count(model)}", flush=True)

# Load checkpoint if exists
round_num, round_history = load_checkpoint(model)
if round_num > 0:
    print(f"[Node {NODE_ID}] Resuming from round {round_num}, {MAX_ROUNDS - round_num} rounds remaining.", flush=True)

# Thread-safe buffer for received peer weights
received_buffer = []
buffer_lock = threading.Lock()


class DFLServicer(dfl_service_pb2_grpc.DFLServiceServicer):
    def GossipWeights(self, request, context):
        flat = np.frombuffer(request.model_data, dtype=np.float32).copy()
        # Use the sender's sample_count for correct FedAvg weighting.
        peer_sample_count = request.sample_count if request.sample_count > 0 else 1
        with buffer_lock:
            received_buffer.append((flat, peer_sample_count))
        print(
            f"[Node {NODE_ID}] Received weights from Node {request.node_id} "
            f"({len(flat)} params, {peer_sample_count} samples). Buffer size: {len(received_buffer)}",
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

        # Step 1: Train and evaluate on val set
        loss, acc = train_local(model, train_data, val_data, epochs=LOCAL_EPOCHS)
        if acc is not None:
            round_history.append((round_num, loss, acc))

        # Step 2: FedAvg with any peer weights received this round
        with buffer_lock:
            snapshot = received_buffer.copy()
            received_buffer.clear()

        if snapshot:
            local_flat = np.concatenate(
                [w.flatten() for w in model.get_weights()]
            ).astype(np.float32)

            local_param_count = len(local_flat)
            compatible = [(f, n) for f, n in snapshot if len(f) == local_param_count]
            skipped = len(snapshot) - len(compatible)

            if skipped:
                print(
                    f"[Node {NODE_ID}] Skipped {skipped} peer weight(s) — "
                    f"incompatible architecture (different num_classes).",
                    flush=True,
                )

            if compatible:
                aggregated = fedavg(local_flat, local_n, compatible)
                shapes = [w.shape for w in model.get_weights()]
                new_weights = []
                offset = 0
                for shape in shapes:
                    size = int(np.prod(shape))
                    new_weights.append(aggregated[offset:offset + size].reshape(shape))
                    offset += size
                model.set_weights(new_weights)
                print(f"[Node {NODE_ID}] Round {round_num}: FedAvg applied ({len(compatible)} peers).", flush=True)
            else:
                print(f"[Node {NODE_ID}] Round {round_num}: No compatible peer weights — keeping local model.", flush=True)
        else:
            print(f"[Node {NODE_ID}] Round {round_num}: No peer weights — keeping local model.", flush=True)

        # Step 3: Save checkpoint after every round
        save_checkpoint(model, round_num, round_history)

        # Step 4: Gossip to ALL neighbors every round
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
                    print(f"[Node {NODE_ID}]   {r:3d} | {l:.4f}   | {a * 100:.2f}%{marker}", flush=True)
                print(f"[Node {NODE_ID}] Peak val accuracy: {best[2] * 100:.2f}% at Round {best[0]}", flush=True)
            else:
                print(f"[Node {NODE_ID}] No training data — no accuracy report.", flush=True)

            export_model(model, round_history, train_data)
            server.stop(0)
            break


if __name__ == "__main__":
    serve()