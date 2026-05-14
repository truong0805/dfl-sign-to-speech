import os
import sys
import time
import json
import grpc
import random
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
GOSSIP_INTERVAL = int(os.environ.get("GOSSIP_INTERVAL", 30))   # seconds between rounds
LOCAL_EPOCHS    = int(os.environ.get("LOCAL_EPOCHS", 1))        # epochs per FL round
MAX_ROUNDS      = int(os.environ.get("MAX_ROUNDS", 0))          # 0 = run forever


def load_sample_count():
    meta_path = os.path.join(DATA_DIR, "metadata.json")
    try:
        with open(meta_path) as f:
            return json.load(f).get("sample_count", 1)
    except Exception as e:
        print(f"[Node {NODE_ID}] WARNING: Could not load metadata.json: {e}. Using 1.", flush=True)
        return 1


def load_training_data():
    """Load this node's local image shard via flow_from_directory."""
    datagen = ImageDataGenerator(rescale=1.0 / 255)
    try:
        data = datagen.flow_from_directory(
            DATA_DIR,
            target_size=(128, 128),
            batch_size=16,
            class_mode="categorical",
        )
        print(f"[Node {NODE_ID}] Loaded {data.samples} images, {data.num_classes} classes.", flush=True)
        return data
    except Exception as e:
        print(f"[Node {NODE_ID}] WARNING: Could not load training data: {e}. Skipping training.", flush=True)
        return None


def train_local(model, data, epochs=1):
    """Train the model on this node's local data shard. Returns (loss, acc) or (None, None)."""
    if data is None or data.samples == 0:
        print(f"[Node {NODE_ID}] No local data — skipping training.", flush=True)
        return None, None
    print(f"[Node {NODE_ID}] Training {epochs} epoch(s) on {data.samples} samples...", flush=True)
    history = model.fit(data, epochs=epochs, verbose=0)
    acc = history.history.get("accuracy", [0])[-1]
    loss = history.history.get("loss", [0])[-1]
    print(f"[Node {NODE_ID}] Local train done. loss={loss:.4f}  acc={acc * 100:.2f}%", flush=True)
    return loss, acc


def fedavg(local_flat, local_n, peer_weights):
    """Weighted FedAvg: w_new = sum(n_i * w_i) / sum(n_i)"""
    all_weights = [(local_flat, local_n)] + peer_weights
    total_n = sum(n for _, n in all_weights)
    if total_n == 0:
        return local_flat
    return (sum(w * n for w, n in all_weights) / total_n).astype(np.float32)


# --- Load data first so we know num_classes, then build matching model ---
local_n = load_sample_count()
print(f"[Node {NODE_ID}] Local sample count: {local_n}", flush=True)

train_data = load_training_data()
num_classes = train_data.num_classes if (train_data and train_data.num_classes > 0) else 36

model = build_model(num_classes=num_classes)
print(f"[Node {NODE_ID}] Model built. Classes: {num_classes}  Total weights: {get_weight_count(model)}", flush=True)

# Buffer of (flat_array, peer_n) tuples received since last aggregation round
received_buffer = []


class DFLServicer(dfl_service_pb2_grpc.DFLServiceServicer):
    def GossipWeights(self, request, context):
        flat = np.frombuffer(request.model_data, dtype=np.float32).copy()
        received_buffer.append((flat, local_n))
        print(
            f"[Node {NODE_ID}] Received weights from Node {request.node_id} "
            f"({len(flat)} params). Buffer size: {len(received_buffer)}",
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

    round_num = 0
    round_history = []  # list of (round, loss, acc)

    while True:
        time.sleep(GOSSIP_INTERVAL)
        round_num += 1
        print(f"[Node {NODE_ID}] ===== FL Round {round_num}{f'/{MAX_ROUNDS}' if MAX_ROUNDS else ''} =====", flush=True)

        # Step 1: Train on local data shard
        loss, acc = train_local(model, train_data, epochs=LOCAL_EPOCHS)
        if acc is not None:
            round_history.append((round_num, loss, acc))

        # Step 2: Apply FedAvg if any weights arrived from peers this round
        if received_buffer:
            local_flat = np.concatenate(
                [w.flatten() for w in model.get_weights()]
            ).astype(np.float32)

            local_param_count = len(local_flat)
            compatible = [(f, n) for f, n in received_buffer if len(f) == local_param_count]
            skipped = len(received_buffer) - len(compatible)
            received_buffer.clear()

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

        # Step 3: Gossip updated weights to a random neighbor
        if NEIGHBORS:
            target = random.choice(NEIGHBORS)
            try:
                with grpc.insecure_channel(f"{target}:50051", options=grpc_options) as channel:
                    stub = dfl_service_pb2_grpc.DFLServiceStub(channel)
                    payload = serialize_weights(model)
                    response = stub.GossipWeights(
                        dfl_service_pb2.WeightRequest(
                            node_id=NODE_ID,
                            model_data=payload,
                        )
                    )
                    if response.success:
                        print(
                            f"[Node {NODE_ID}] Round {round_num}: Gossiped to {target} ({len(payload)} bytes).",
                            flush=True,
                        )
            except Exception as e:
                print(f"[Node {NODE_ID}] Could not reach {target}: {e}", flush=True)

        # Stop and print report when MAX_ROUNDS reached
        if MAX_ROUNDS and round_num >= MAX_ROUNDS:
            print(f"\n[Node {NODE_ID}] ===== TRAINING COMPLETE ({MAX_ROUNDS} rounds) =====", flush=True)
            if round_history:
                best = max(round_history, key=lambda x: x[2])
                print(f"[Node {NODE_ID}] Round | Loss   | Accuracy", flush=True)
                print(f"[Node {NODE_ID}] ------+--------+---------", flush=True)
                for r, l, a in round_history:
                    marker = " <-- PEAK" if r == best[0] else ""
                    print(f"[Node {NODE_ID}]   {r:3d} | {l:.4f} | {a * 100:.2f}%{marker}", flush=True)
                print(f"[Node {NODE_ID}] Peak accuracy: {best[2] * 100:.2f}% at Round {best[0]}", flush=True)
            else:
                print(f"[Node {NODE_ID}] No training data — no accuracy report.", flush=True)
            server.stop(0)
            break


if __name__ == "__main__":
    serve()
