import os
import time
import grpc
import random
import json
import numpy as np
from concurrent import futures

from model import build_model
from train import train_local_model
import dfl_service_pb2
import dfl_service_pb2_grpc
from aggregator import federated_average

class DFLServicer(dfl_service_pb2_grpc.DFLServiceServicer):
    def __init__(self, node_id, sample_count):
        self.node_id = node_id
        self.sample_count = sample_count
        self.received_updates = [] # Buffer for neighbor weights

    def GossipWeights(self, request, context):
        # Deserialization: NumPy from bytes
        weights = np.frombuffer(request.model_data, dtype=np.float32)

        print(f">>> [SERVER] Node {self.node_id} received weights from Node {request.node_id} (n={request.sample_count})", flush=True)

        # Store for Federated Averaging
        self.received_updates.append({
            "weights": weights,
            "sample_count": request.sample_count
        })

        # Trigger aggregation if we have enough neighbors
        if len(self.received_updates) >= 2:
            print(f">>> [SYSTEM] Node {self.node_id} triggering FedAvg aggregation...", flush=True)
            # Logic to merge weights goes here in Week 2/3
            self.received_updates = [] # Clear buffer after use

        return dfl_service_pb2.WeightResponse(success=True)

def serve():
    node_id = os.getenv('NODE_ID')
    neighbors_env = os.getenv('NEIGHBORS', "")
    neighbors = [n.strip() for n in neighbors_env.split(',') if n.strip()]

    # 1. Load Local Metadata (The n value for FedAvg)
    try:
        with open('/app/data/metadata.json', 'r') as f:
            meta = json.load(f)
            sample_count = meta['sample_count']
    except:
        sample_count = 1 # Fallback

    # 2. Start the Server
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    servicer = DFLServicer(node_id, sample_count)
    dfl_service_pb2_grpc.add_DFLServiceServicer_to_server(servicer, server)
    server.add_insecure_port('[::]:50051')
    server.start()
    print(f"--- Node {node_id} online (n={sample_count}) ---", flush=True)

    # 3. Training & Gossip Loop
    while True:
        print(f"--- Node {node_id} starting local training round... ---", flush=True)
        # Call Danh's training function on the local data shard
        local_weights = train_local_model()

        if neighbors:
            target = random.choice(neighbors)
            try:
                with grpc.insecure_channel(f'{target}:50051') as channel:
                    stub = dfl_service_pb2_grpc.DFLServiceStub(channel)

                    # Flatten weights for gRPC transport
                    flattened_weights = np.concatenate([w.flatten() for w in local_weights]).astype(np.float32)

                    response = stub.GossipWeights(dfl_service_pb2.WeightRequest(
                        node_id=int(node_id),
                        model_data=flattened_weights.tobytes(),
                        sample_count=sample_count
                    ))
                    if response.success:
                        print(f"[CLIENT] Node {node_id} gossiped real weights to {target}", flush=True)
            except Exception as e:
                print(f"[CLIENT] Node {node_id} couldn't reach {target}: {e}", flush=True)

        time.sleep(30) # Wait between rounds

if __name__ == '__main__':
    serve()