import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from model_image import build_model, fine_tune_model
from utils import get_weights
import json

DATA_DIR = os.environ.get("IMAGE_DATA_DIR", "resized_dataset")

# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------
# IMPORTANT: horizontal_flip is DISABLED.
# ASL signs are not mirror-symmetric — many letters (J, Z, G, H, etc.) have
# directional handshapes or motions that are meaningless or map to a different
# letter when flipped. Flipping creates corrupted training samples.
#
# Augmentations kept are safe for ASL:
#   - Small rotation (±10°): natural hand tilt variation
#   - Small shift (10%): hand not always perfectly centered
#   - Zoom (10%): distance to camera varies
#   - Brightness (±20%): lighting variation across environments
# ---------------------------------------------------------------------------
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    brightness_range=[0.8, 1.2],   # simulate different lighting conditions
    fill_mode='nearest',
)

val_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2,
)

train_data = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    subset='training',
    shuffle=True,
)

val_data = val_datagen.flow_from_directory(
    DATA_DIR,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    subset='validation',
    shuffle=False,
)

print(f"Training samples : {train_data.samples}")
print(f"Validation samples: {val_data.samples}")
print(f"Classes          : {train_data.num_classes}")

# ---------------------------------------------------------------------------
# Callbacks (shared between both phases)
# ---------------------------------------------------------------------------
early_stop = EarlyStopping(
    monitor='val_accuracy',
    patience=3,
    restore_best_weights=True,
    verbose=1,
)
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=2,
    min_lr=1e-6,
    verbose=1,
)

# ---------------------------------------------------------------------------
# Phase 1: Train head only (base frozen)
# ---------------------------------------------------------------------------
print("\n===== Phase 1: Training head (base frozen) =====")
model = build_model(num_classes=train_data.num_classes)

model.fit(
    train_data,
    epochs=10,
    validation_data=val_data,
    callbacks=[early_stop, reduce_lr],
)

loss, acc = model.evaluate(val_data, verbose=0)
print(f"Phase 1 val accuracy: {acc * 100:.2f}%")

# ---------------------------------------------------------------------------
# Phase 2: Fine-tune top MobileNetV2 layers
# ---------------------------------------------------------------------------
print("\n===== Phase 2: Fine-tuning top 30 MobileNetV2 layers =====")
model = fine_tune_model(model, num_unfreeze=30)

# Reset early stopping patience for phase 2
early_stop_ft = EarlyStopping(
    monitor='val_accuracy',
    patience=5,                     # more patience — fine-tuning is slower
    restore_best_weights=True,
    verbose=1,
)
reduce_lr_ft = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-7,
    verbose=1,
)

model.fit(
    train_data,
    epochs=15,
    validation_data=val_data,
    callbacks=[early_stop_ft, reduce_lr_ft],
)

loss, acc = model.evaluate(val_data, verbose=0)
print(f"Phase 2 val accuracy: {acc * 100:.2f}%")

# ---------------------------------------------------------------------------
# Save weights for DFL node loading
# ---------------------------------------------------------------------------
with open("weights_image.json", "w") as f:
    json.dump(get_weights(model), f)

print("Saved weights_image.json")