# DFL Sign-to-Speech

> **Decentralized Federated Learning for American Sign Language Recognition**
>
> VGU · Year 3 · Semester 2 · Distributed Systems Project

A privacy-preserving distributed machine learning system where 5 autonomous Docker nodes collaboratively train a CNN model to recognize ASL hand signs — without ever sharing raw image data. Recognized signs are converted to speech via text-to-speech output.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [System Flow](#system-flow)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Team & Responsibilities](#team--responsibilities)
- [Getting Started](#getting-started)
- [Configuration](#configuration)
- [Data Pipeline](#data-pipeline)
- [Model Details](#model-details)
- [gRPC Protocol](#grpc-protocol)
- [Current Status](#current-status)

---

## Overview

Traditional machine learning requires centralizing all training data on a single server — a privacy risk when the data is sensitive. This project implements **Decentralized Federated Learning (DFL)**: each of 5 nodes trains on its own private image shard, then periodically exchanges only model *weights* (not images) with peers via a gossip protocol.

**Key properties:**

- **No raw data sharing** — images stay on the node that owns them
- **Non-IID data** — each node sees a different subset of ASL classes (realistic distribution)
- **Fault-tolerant** — nodes gossip asynchronously; one slow or missing peer does not block others
- **Horizontally scalable** — adding more nodes requires no architectural changes

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Docker Network (dfl-net)                  │
│                                                             │
│   ┌────────┐     gRPC      ┌────────┐     gRPC      ┌────────┐
│   │ node1  │◄─────────────►│ node3  │◄─────────────►│ node5  │
│   │  CNN   │               │  CNN   │               │  CNN   │
│   │ shard1 │               │ shard3 │               │ shard5 │
│   └────────┘               └────────┘               └────────┘
│       ▲  ▲                     ▲                        ▲
│       │  └─────────────────────┤                        │
│   ┌────────┐     gRPC      ┌────────┐                   │
│   │ node2  │◄─────────────►│ node4  │◄──────────────────┘
│   │  CNN   │               │  CNN   │
│   │ shard2 │               │ shard4 │
│   └────────┘               └────────┘
│                                                             │
│   Full mesh — each node can reach all 4 peers              │
└─────────────────────────────────────────────────────────────┘
```

Each node is a Python process that runs:

1. A **gRPC server** (port 50051) to receive weight gossip from peers
2. A **training loop** to fine-tune its local CNN model on its private data shard
3. A **gossip client** to push its weights to a random peer each round

---

## System Flow

```
┌──────────────────────────────────────────────────────────┐
│  Per-node Federated Learning Round                       │
│                                                          │
│  1. Load local data shard                                │
│     └── metadata.json → sample_count (n_i)               │
│                                                          │
│  2. Train locally (LOCAL_EPOCHS)                         │
│     └── Keras CNN on private ASL images                  │
│                                                          │
│  3. Serialize weights                                    │
│     └── flatten all layers → np.float32 → bytes         │
│                                                          │
│  4. Gossip to random peer                                │
│     └── gRPC GossipWeights RPC → WeightRequest           │
│                                                          │
│  5. Receive weights from peers                           │
│     └── buffered by gRPC server in background            │
│                                                          │
│  6. FedAvg aggregation                                   │
│     └── w_global = Σ(n_i × w_i) / Σ(n_i)               │
│                                                          │
│  7. Apply aggregated weights to local model              │
│                                                          │
│  8. Sleep GOSSIP_INTERVAL → repeat                       │
└──────────────────────────────────────────────────────────┘
```

### FedAvg Formula

Weighted average by number of training samples, following the original McMahan et al. (2017) paper:

```
w_global = ( n_1×w_1 + n_2×w_2 + ... + n_k×w_k ) / ( n_1 + n_2 + ... + n_k )
```

where `n_i` is the sample count on node `i` and `w_i` is its serialized weight vector.

---

## Tech Stack

| Layer | Technology | Version |
|---|---|---|
| Language | Python | 3.10 |
| ML Framework | TensorFlow / Keras | ≥ 2.12 (CPU) |
| Communication | gRPC + Protocol Buffers | grpcio |
| Serialization | NumPy float32 bytes | — |
| Containerization | Docker + Docker Compose | — |
| Image Processing | Pillow | — |
| TTS (planned) | gTTS / pyttsx3 | — |
| UI (planned) | Streamlit | — |

---

## Project Structure

```
dfl-sign-to-speech/
│
├── app/                            # Runtime: gRPC server + FL loop
│   ├── node.py                     # Entry point — gRPC server & gossip client
│   ├── model.py                    # Model abstraction (build/serialize/deserialize)
│   ├── utils.py                    # TTS integration (placeholder)
│   ├── dfl_service_pb2.py          # Auto-generated proto message classes
│   └── dfl_service_pb2_grpc.py     # Auto-generated gRPC service stubs
│
├── DistributedSystemProject/       # ML models & standalone training scripts
│   ├── model_mnist.py              # CNN for 28×28 grayscale MNIST (24 classes)
│   ├── model_image.py              # CNN for 128×128 RGB images (36 classes)
│   ├── train_mnist.py              # Standalone MNIST training
│   ├── train_image.py              # Standalone image model training
│   └── utils.py                    # Weight extraction utilities
│
├── scripts/
│   └── partition.py                # Non-IID data sharding across 5 nodes
│
├── protos/
│   └── dfl_service.proto           # gRPC service contract
│
├── data_shards/                    # Partitioned training data (git-ignored)
│   ├── node1/
│   │   ├── metadata.json           # { node_id, sample_count, labels: {...} }
│   │   └── [class]/[images]
│   ├── node2/ ... node5/
│
├── resized_dataset/                # Preprocessed 128×128 images
│
├── resize.py                       # Image preprocessing (→ 128×128 RGB)
├── docker-compose.yml              # 5-node cluster definition
├── Dockerfile                      # Python 3.10-slim image
└── requirements.txt                # Python dependencies
```

---

## Team & Responsibilities

| Member | Role | Files | Status |
|---|---|---|---|
| **Nhan** | P2P & Orchestration | `app/node.py`, `protos/` | ✅ Functional |
| **Danh** | Model & Training | `DistributedSystemProject/model_*.py`, `train_*.py` | ✅ Done |
| **Nguyen** | Data Pipeline | `scripts/partition.py`, `resize.py`, `data_shards/` | ✅ Done |
| **Khang** | Docker & Networking | `docker-compose.yml`, `Dockerfile` | ✅ Done |
| **Phuc** | UI & TTS | `app/utils.py`, Streamlit UI | ⏳ In Progress |

---

## Getting Started

### Prerequisites

- Docker Desktop (with Compose v2)
- Python 3.10+ (for local dev / standalone training only)
- ~2 GB disk space for dataset

### 1. Clone the repository

```bash
git clone <repo-url>
cd dfl-sign-to-speech
```

### 2. Prepare the dataset

Place raw ASL images organized by class folder under `raw_dataset/`:

```
raw_dataset/
├── A/  ├── img1.jpg  └── ...
├── B/  └── ...
...
```

Then preprocess and partition:

```bash
# Resize all images to 128×128 RGB
python resize.py

# Partition into 5 non-IID shards
python scripts/partition.py
```

This creates `data_shards/node1` through `data_shards/node5`, each with a `metadata.json`.

### 3. Launch the cluster

```bash
docker-compose up --build
```

All 5 nodes start in parallel. Each waits for its gRPC server to bind, then begins the FL training loop.

### 4. Monitor logs

```bash
# Tail all nodes
docker-compose logs -f

# Tail a specific node
docker-compose logs -f node1
```

You should see output like:

```
[Node 1] ===== FL Round 1/10 =====
[Node 1] Training 3 epoch(s) on 42 samples...
[Node 1] Local train done. loss=0.8234  acc=78.45%
[Node 1] Round 1: Gossiped to node2 (182400 bytes).
[Node 1] Round 1: FedAvg applied (2 peers).
```

### 5. Stop the cluster

```bash
docker-compose down
```

### Standalone model training (optional)

To train a model locally without Docker:

```bash
pip install -r requirements.txt

python DistributedSystemProject/train_mnist.py
# or
python DistributedSystemProject/train_image.py
```

### Regenerate gRPC stubs (after proto changes)

```bash
python -m grpc_tools.protoc \
  -I protos \
  --python_out=app \
  --grpc_python_out=app \
  protos/dfl_service.proto
```

---

## Configuration

All per-node settings are injected via environment variables in `docker-compose.yml`:

| Variable | Default | Description |
|---|---|---|
| `NODE_ID` | `1` | Unique identifier for this node (1–5) |
| `NEIGHBORS` | `""` | Comma-separated hostnames of peer nodes |
| `LOCAL_EPOCHS` | `3` | Epochs to train locally per FL round |
| `GOSSIP_INTERVAL` | `30` | Seconds to sleep between rounds |
| `MAX_ROUNDS` | `10` | Total FL rounds to run (0 = infinite) |

Example for node1 in `docker-compose.yml`:

```yaml
node1:
  build: .
  volumes:
    - ./app:/app
    - ./data_shards/node1:/app/data
  environment:
    - NODE_ID=1
    - NEIGHBORS=node2,node3,node4,node5
    - LOCAL_EPOCHS=3
    - GOSSIP_INTERVAL=30
    - MAX_ROUNDS=10
```

---

## Data Pipeline

### Non-IID Sharding (`scripts/partition.py`)

Real-world federated settings have non-identical data distributions across clients. This script simulates that:

1. Load all class folders from `resized_dataset/`
2. For each class: split images into 10 random shards
3. Shuffle all shards globally
4. Assign 6 shards to each of the 5 nodes (round-robin)
5. Write `metadata.json` per node

Example `data_shards/node1/metadata.json`:

```json
{
  "node_id": 1,
  "sample_count": 42,
  "labels": {
    "d": 7,
    "h": 7,
    "j": 14,
    "o": 7,
    "w": 7
  }
}
```

Node 1 only sees 5 of the 36 possible ASL classes — forcing the global model to learn from all classes through weight aggregation.

---

## Model Details

### Image CNN (`model_image.py`)

Trained on 128×128 RGB images, classifying 36 ASL signs (A–Z + 0–9).

```
Input: (128, 128, 3)
  → Conv2D(32, 3×3, relu)
  → Conv2D(64, 3×3, relu)
  → MaxPooling2D
  → Conv2D(128, 3×3, relu)
  → MaxPooling2D
  → Flatten
  → Dense(128, relu)
  → Dense(36, softmax)
```

### MNIST CNN (`model_mnist.py`)

Trained on 28×28 grayscale images, classifying 24 ASL letters (A–X, excluding J and Z which require motion).

```
Input: (28, 28, 1)
  → Conv2D(32, 3×3, relu) → MaxPooling2D
  → Conv2D(64, 3×3, relu) → MaxPooling2D
  → Flatten
  → Dense(128, relu)
  → Dense(24, softmax)
```

### Weight Serialization

Weights cross the network as raw bytes — no framework-specific format, no overhead:

```python
# Serialize (sender)
flat = np.concatenate([w.flatten() for w in model.get_weights()]).astype(np.float32)
payload = flat.tobytes()

# Deserialize (receiver)
flat = np.frombuffer(data, dtype=np.float32).copy()
# reshape layer-by-layer and call model.set_weights(...)
```

---

## gRPC Protocol

Defined in `protos/dfl_service.proto`:

```protobuf
syntax = "proto3";

service DFLService {
  rpc GossipWeights (WeightRequest) returns (WeightResponse);
}

message WeightRequest {
  int32 node_id  = 1;   // Sender node ID
  bytes model_data = 2; // Serialized np.float32 weight vector
}

message WeightResponse {
  bool success = 1;
}
```

All nodes listen on port **50051**. The gRPC server runs in a background thread while the FL loop runs on the main thread.

---

## Current Status

| Component | Status | Notes |
|---|---|---|
| gRPC server / gossip client | ✅ Functional | Sends and receives weight bytes |
| FedAvg aggregation | ✅ Functional | Weighted average over buffered peers |
| Image CNN (128×128) | ✅ Functional | Trains standalone |
| MNIST CNN (28×28) | ✅ Functional | Trains standalone |
| Data sharding | ✅ Complete | 5 non-IID shards with metadata |
| Docker cluster | ✅ Functional | All 5 nodes boot and mesh correctly |
| Model → gRPC integration | ⏳ In Progress | Real weights not yet wired to gossip |
| TTS output | ⏳ Planned | `app/utils.py` is a placeholder |
| Streamlit UI | ⏳ Planned | Week 4 milestone |
| Model checkpointing | ⏳ Planned | Save/load weights between rounds |

---

## References

- McMahan et al., *Communication-Efficient Learning of Deep Networks from Decentralized Data* (FedAvg), 2017
- gRPC Python documentation: [grpc.io](https://grpc.io/docs/languages/python/)
- ASL MNIST dataset: [Kaggle](https://www.kaggle.com/datasets/datamunge/sign-language-mnist)
- TensorFlow / Keras documentation: [tensorflow.org](https://www.tensorflow.org/)
