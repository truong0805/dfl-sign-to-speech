import os
import time
import grpc
import random
import numpy as np
from concurrent import futures
import dfl_service_pb2
import dfl_service_pb2_grpc
import json

class DFLServicer(dfl_service_pb2_grpc.DFLServiceServicer):
    def __init__(self):
        self.received_updates = [] # Buffer to hold weights from neighbors

    def GossipWeights(self, request, context):
        # Deserialization
        weights = np.frombuffer(request.model_data, dtype=np.float32)

        self.received_updates.append({
            "node_id": request.node_id,
            "weights": weights,
            "sample_count": request.sample_count # You'll need to add this to your .proto!
        })

        # Trigger aggregation if we have enough neighbors (e.g., 2 neighbors)
        if len(self.received_updates) >= 2:
            print(">>> [SYSTEM] Sufficient updates received. Triggering FedAvg...")
            self.received_updates = [] # Clear the buffer so you don't re-trigger too fast
            # Here is where you will call aggregator.federated_average()

        return dfl_service_pb2.WeightResponse(success=True)

def serve():
    node_id = os.getenv('NODE_ID')
    neighbors_env = os.getenv('NEIGHBORS', "")
    neighbors = [n.strip() for n in neighbors_env.split(',') if n.strip()]

    # 1. Start the Server (The "Ear")
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    dfl_service_pb2_grpc.add_DFLServiceServicer_to_server(DFLServicer(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    print(f"--- Node {node_id} online, neighbors: {neighbors} ---", flush=True)

    # 2. Client Loop (The "Mouth")
    while True:
        if neighbors:
            target = random.choice(neighbors) # P2P Gossip logic
            try:
                with grpc.insecure_channel(f'{target}:50051') as channel:
                    stub = dfl_service_pb2_grpc.DFLServiceStub(channel)

                    # Create dummy NumPy weights
                    dummy_weights = np.random.rand(10).astype(np.float32)

                    response = stub.GossipWeights(dfl_service_pb2.WeightRequest(
                        node_id=int(node_id),
                        model_data=dummy_weights.tobytes() # Serialize to bytes
                    ))
                    if response.success:
                        print(f"[CLIENT] Node {node_id} successfully gossiped to {target}", flush=True)
            except Exception:
                print(f"[CLIENT] Node {node_id} couldn't reach {target}. Retrying...", flush=True)

        time.sleep(10)

if __name__ == '__main__':
    serve()

def get_local_metadata():
    try:
        # This path matches the volume mount we set in docker-compose
        with open('/app/data/metadata.json', 'r') as f:
            data = json.load(f)
            return data['sample_count']
    except Exception as e:
        print(f"Error loading metadata: {e}")
        return 1 # Default to 1 to avoid division by zero