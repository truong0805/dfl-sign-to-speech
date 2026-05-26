from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam


def build_model(input_shape=(128, 128, 3), num_classes=26):
    """
    Phase 1 model: MobileNetV2 base fully frozen.
    Only the 4-layer custom head is trained.
    Call fine_tune_model() afterwards for Phase 2.
    """
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    # Phase 1: freeze entire base
    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),   # wider than before (128→256)
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=Adam(learning_rate=1e-3),     # higher LR is fine when base is frozen
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def fine_tune_model(model, num_unfreeze=30):
    """
    Phase 2: unfreeze the top `num_unfreeze` layers of MobileNetV2
    and recompile with a much lower learning rate.
    Call this after Phase 1 training is complete.

    MobileNetV2 has 154 layers total; unfreezing the last 30 gives the
    model enough flexibility to learn ASL-specific hand features while
    keeping the early edge/texture detectors frozen (stable, fast).
    """
    base_model = model.layers[0]               # the MobileNetV2 sub-model
    base_model.trainable = True

    # Re-freeze everything except the last `num_unfreeze` layers
    for layer in base_model.layers[:-num_unfreeze]:
        layer.trainable = False

    trainable_count = sum(1 for l in base_model.layers if l.trainable)
    print(f"[fine_tune_model] Unfroze top {trainable_count} MobileNetV2 layers.")

    model.compile(
        optimizer=Adam(learning_rate=1e-4),     # 10× lower than Phase 1
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model