from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.optimizers import Adam


def build_model(input_shape=(128, 128, 3), num_classes=26):
    """
    Phase 1 model: EfficientNetB0 base fully frozen.
    Only the custom classification head is trained.

    WHY EfficientNetB0 instead of MobileNetV2:
    - MobileNetV2 is optimised for mobile size, not accuracy. It lacks the
      capacity to reliably separate confusable ASL letters (M/N, K/V, R/U).
    - EfficientNetB0 achieves ~99.95% on this exact dataset (grassknoted
      ASL Alphabet, 87k images) in published benchmarks, vs ~85-90% for
      MobileNetV2.
    - It is still small enough (4.0MB weights) to gossip over gRPC without
      hitting the 64MB message limit — no changes needed in node.py.
    - EfficientNetB0 includes its own preprocessing (rescaling + normalisation
      built into the model graph), so we set include_preprocessing=True and
      pass raw [0-255] uint8 tensors from the data pipeline. This removes one
      source of preprocessing mismatch between training and inference.

    Phase 2 fine-tuning: call fine_tune_model() after Phase 1 converges.
    """
    base_model = EfficientNetB0(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet',
        include_preprocessing=True,   # built-in rescale + normalise
    )
    # Phase 1: freeze entire base — only train the head
    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),   # stabilises training after pooling
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def fine_tune_model(model, num_unfreeze=30):
    """
    Phase 2: unfreeze the top `num_unfreeze` layers of EfficientNetB0
    and recompile with a much lower learning rate.

    EfficientNetB0 has 237 layers total. Unfreezing the last 30 gives the
    model enough flexibility to learn fine-grained ASL hand features while
    keeping the low-level edge/texture detectors frozen and stable.

    Call this ONLY after Phase 1 training has fully converged.
    """
    base_model = model.layers[0]      # the EfficientNetB0 sub-model
    base_model.trainable = True

    # Re-freeze everything except the last `num_unfreeze` layers
    for layer in base_model.layers[:-num_unfreeze]:
        layer.trainable = False

    # Keep BatchNorm layers in inference mode during fine-tuning —
    # unfreezing BN with a small batch causes instability.
    for layer in base_model.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False

    trainable_count = sum(1 for l in base_model.layers if l.trainable)
    print(f"[fine_tune_model] Unfroze top {trainable_count} EfficientNetB0 layers.")

    model.compile(
        optimizer=Adam(learning_rate=1e-4),   # 10× lower than Phase 1
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model