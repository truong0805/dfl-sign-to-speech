"""
model_image.py — Optimized MobileNetV2 backbone for Sign-to-Speech DFL.

CHANGES FROM PREVIOUS VERSION:
  - Added Dense(256, relu) intermediate layer before final classifier.
    Gives the head more capacity to separate visually similar ASL signs
    (e.g. R/U, M/N, S/E) without significantly increasing param count.
  - Replaced fixed Adam(1e-3) with CosineDecayRestarts schedule.
    Prevents loss oscillation across FL rounds caused by a constant LR
    that never settles the weights after each FedAvg merge.
  - Dropout raised from 0.2 → 0.3 to compensate for the added Dense layer.
  - fine_tune_model() unchanged — called by node.py at FINE_TUNE_ROUND.
"""

import tensorflow as tf
from tensorflow.keras import layers, models, Input
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import CosineDecayRestarts

INPUT_SHAPE = (128, 128, 3)

# One "round" ≈ 4000 images / batch_size=8 = 500 steps, times LOCAL_EPOCHS=5 = 2500 steps.
# CosineDecayRestarts restarts every first_decay_steps — set to ~1 round of steps.
# This lets the LR warm-restart after each FedAvg merge, which helps convergence.
_FIRST_DECAY_STEPS = 2500
_INITIAL_LR        = 1e-3
_MIN_LR            = 1e-5


def _make_lr_schedule(initial_lr=_INITIAL_LR,
                      first_decay_steps=_FIRST_DECAY_STEPS,
                      min_lr=_MIN_LR):
    return CosineDecayRestarts(
        initial_learning_rate=initial_lr,
        first_decay_steps=first_decay_steps,
        t_mul=1.0,
        m_mul=1.0,
        alpha=min_lr,
    )


def build_model(input_shape=INPUT_SHAPE, num_classes=26):
    """
    Phase 1: MobileNetV2 base fully frozen, train classification head only.

    Head: GAP → BatchNorm → Dense(256, relu) → Dropout(0.3) → Dense(26, softmax)
    The extra Dense(256) layer gives the classifier capacity to distinguish
    visually close ASL signs without fine-tuning the backbone.

    Preprocessing: Rescaling [0, 255] → [-1, 1] is inside the graph so the
    data pipeline never needs to do manual normalisation.
    """
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet',
        alpha=1.0,
    )
    base_model.trainable = False  # Phase 1: freeze entire backbone

    inputs = Input(shape=input_shape, name='image_input')

    # In-graph preprocessing: [0, 255] → [-1, 1] as MobileNetV2 expects
    x = layers.Rescaling(scale=1.0 / 127.5, offset=-1.0, name='preprocess')(inputs)

    # Frozen feature extraction
    x = base_model(x, training=False)   # keeps BatchNorm in inference mode
    x = layers.GlobalAveragePooling2D(name='gap')(x)
    x = layers.BatchNormalization(name='head_bn')(x)

    # Intermediate layer — helps separate visually similar hand shapes
    x = layers.Dense(256, activation='relu', name='head_dense')(x)
    x = layers.Dropout(0.3, name='head_dropout')(x)

    outputs = layers.Dense(num_classes, activation='softmax', name='predictions')(x)

    model = models.Model(inputs, outputs, name='mobilenetv2_asl')

    model.compile(
        optimizer=Adam(_make_lr_schedule()),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )

    print(f"[build_model] Optimized MobileNetV2 | input={input_shape} | classes={num_classes}")
    print(f"[build_model] Total params: {model.count_params():,}")
    return model


def fine_tune_model(model, num_unfreeze=30):
    """
    Phase 2: unfreeze the top `num_unfreeze` layers of the MobileNetV2 backbone.

    Called by node.py when round_num >= FINE_TUNE_ROUND (set to ~20 in docker-compose).

    Changes from Phase 1:
      - Increased default num_unfreeze from 20 → 30 to expose more domain-specific
        feature layers for ASL (hand textures differ significantly from ImageNet).
      - Lower initial LR (1e-5) with tighter cosine decay to avoid destroying
        pretrained features during distributed fine-tuning.
      - BatchNorm layers in the backbone remain frozen — critical for small
        per-node batch sizes (batch=8 gives noisy BN statistics).
    """
    base_model = None
    for layer in model.layers:
        if hasattr(layer, 'layers') and 'mobilenetv2' in layer.name.lower():
            base_model = layer
            break

    if base_model is None:
        print("[fine_tune_model] WARNING: MobileNetV2 sub-model not found — skipping.")
        return model

    base_model.trainable = True

    # Freeze all but the top num_unfreeze layers
    for layer in base_model.layers[:-num_unfreeze]:
        layer.trainable = False

    # Always freeze BatchNorm in the backbone — batch=8 is too small for stable BN
    for layer in base_model.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False

    trainable_count = sum(1 for l in base_model.layers if l.trainable)
    print(f"[fine_tune_model] Unfroze top {trainable_count} MobileNetV2 layers.")

    # Fine-tune schedule: lower LR, tighter decay
    ft_schedule = CosineDecayRestarts(
        initial_learning_rate=1e-5,
        first_decay_steps=_FIRST_DECAY_STEPS,
        t_mul=1.0,
        alpha=1e-7,
    )

    model.compile(
        optimizer=Adam(ft_schedule),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model