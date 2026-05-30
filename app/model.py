"""
model.py — Fixed Bridge Config for Sign-to-Speech DFL.
Ensures correct path resolution inside Docker containers.
"""
import os
import sys
import numpy as np

# Absolute path correction for Docker environment safety
CONTAINER_ROOT = "/app"
PROJECT_SUBDIR = os.path.join(CONTAINER_ROOT, "DistributedSystemProject")

if os.path.exists(PROJECT_SUBDIR):
    if PROJECT_SUBDIR not in sys.path:
        sys.path.insert(0, PROJECT_SUBDIR)
else:
    # Local fallback path configuration
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'DistributedSystemProject'))

from model_image import build_model as _build_image_model

MODEL_TYPE = os.environ.get("MODEL_TYPE", "image")


def build_model(num_classes=26):
    if MODEL_TYPE == "mnist":
        from model_mnist import build_model as _build_mnist_model
        return _build_mnist_model()
    return _build_image_model(input_shape=(160, 160, 3), num_classes=num_classes)


def serialize_weights(model) -> bytes:
    """Flatten all Keras layer weights to a single np.float32 byte string for gRPC transport."""
    flat = np.concatenate([w.flatten() for w in model.get_weights()]).astype(np.float32)
    return flat.tobytes()


def deserialize_weights(model, data: bytes) -> None:
    """Restore Keras model weights from a flat float32 byte string."""
    flat = np.frombuffer(data, dtype=np.float32).copy()
    shapes = [w.shape for w in model.get_weights()]
    weights = []
    offset = 0
    for shape in shapes:
        size = int(np.prod(shape))
        weights.append(flat[offset:offset + size].reshape(shape))
        offset += size
    model.set_weights(weights)


def get_weight_count(model) -> int:
    return sum(w.size for w in model.get_weights())