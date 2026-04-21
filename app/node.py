import os
import time
import grpc
import random
import numpy as np
from concurrent import futures
import dfl_service_pb2
import dfl_service_pb2_grpc

class DFLServicer(dfl_service_pb2_grpc.DFLServiceServicer):
    def GossipWeights(self, request, context):
        received_weights = np.frombuffer(request.model_data, dtype=np.float32) # turn the raw bytes received over the network back into a usable NumPy array

        print(f">>> [SERVER] Node {os.getenv('NODE_ID')} received weights from Node {request.node_id}", flush=True)
        print(f"--- Received Sample: {received_weights[:3]}...", flush=True)

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