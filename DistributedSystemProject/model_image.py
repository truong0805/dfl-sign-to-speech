"""
model_image.py — Optimized MobileNetV2 backbone for Sign-to-Speech DFL.

OPTIMIZATIONS:
  - Streamlined the classification head by connecting GlobalAveragePooling2D
    directly to the final Dense(26) layer via Dropout. This reduces parameter
    density, saves activation RAM, and accelerates local epoch convergence.
  - Retained the functional Keras 3 pattern to avoid input_shape warnings.
"""

from tensorflow.keras import layers, models, regularizers, Input
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam

INPUT_SHAPE = (128, 128, 3)


def build_model(input_shape=INPUT_SHAPE, num_classes=26):
    """
    Phase 1: MobileNetV2 base fully frozen, train streamlined head.
    Preprocesses images automatically from [0, 255] to [-1, 1].
    """
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet',
        alpha=1.0,
    )
    base_model.trainable = False  # Phase 1: freeze entire base

    # Functional API build pattern
    inputs = Input(shape=input_shape, name='image_input')

    # Preprocessing layer: scale [0, 255] → [-1, 1] inside the execution graph
    x = layers.Rescaling(scale=1.0 / 127.5, offset=-1.0, name='preprocess')(inputs)

    # Base feature extraction
    x = base_model(x, training=False)   # Keeps BatchNormalization in inference mode
    x = layers.GlobalAveragePooling2D(name='gap')(x)
    x = layers.BatchNormalization(name='head_bn')(x)

    # Optimized: Removed Dense(128) to flatten computational overhead
    x = layers.Dropout(0.2, name='head_dropout')(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='predictions')(x)

    model = models.Model(inputs, outputs, name='mobilenetv2_asl')

    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    print(f"[build_model] Optimized MobileNetV2 | input={input_shape} | classes={num_classes}")
    print(f"[build_model] Total params: {model.count_params():,}")
    return model


def fine_tune_model(model, num_unfreeze=20):
    """
    Phase 2: unfreeze the top layers of MobileNetV2 for minor refinement.
    """
    base_model = None
    for layer in model.layers:
        if isinstance(layer, MobileNetV2.__class__) or 'mobilenetv2' in layer.name.lower():
            base_model = layer
            break

    if base_model is None:
        from tensorflow.keras.applications import MobileNetV2 as MNV2
        for layer in model.layers:
            if hasattr(layer, 'layers'):
                base_model = layer
                break

    if base_model is None:
        print("[fine_tune_model] WARNING: could not find MobileNetV2 sub-model.")
        return model

    base_model.trainable = True

    for layer in base_model.layers[:-num_unfreeze]:
        layer.trainable = False

    # Freeze BN to handle small decentralized batch restrictions safely
    for layer in base_model.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False

    trainable_count = sum(1 for l in base_model.layers if l.trainable)
    print(f"[fine_tune_model] Unfroze top {trainable_count} MobileNetV2 layers.")

    model.compile(
        optimizer=Adam(learning_rate=1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model