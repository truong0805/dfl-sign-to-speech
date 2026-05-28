"""
train_image.py — standalone full-dataset trainer for MobileNetV2.

Run this ONCE on the full dataset (before DFL) to verify the model can
learn the task. The resulting val_acc gives an upper-bound benchmark for
what the DFL nodes should approach after enough rounds.

Key changes from EfficientNetB0 version:
  - Backbone: EfficientNetB0 → MobileNetV2 (lighter activation RAM, no crashes)
  - IMG_SIZE: kept at 128×128 — matches resize.py, no re-resizing needed
  - Preprocessing: Lambda layer inside model handles [0,255] → [-1,1]
                   so data pipeline stays the same (cast to float32, no division)
"""

import os
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from model_image import build_model, fine_tune_model
import json

DATA_DIR   = os.environ.get("IMAGE_DATA_DIR", "resized_dataset")
AUTOTUNE   = tf.data.AUTOTUNE
IMG_SIZE   = (128, 128)    # matches resize.py — no re-resizing needed
BATCH_SIZE = 32            # safe for standalone training on 16 GB machine


def load_dataset(split_dir, shuffle):
    ds = tf.keras.utils.image_dataset_from_directory(
        split_dir,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='categorical',
        shuffle=shuffle,
    )
    # Cast to float32 in [0, 255].
    # The Lambda preprocess layer inside build_model() converts to [-1, 1]
    # as MobileNetV2 requires — no manual division here.
    ds = ds.map(lambda x, y: (tf.cast(x, tf.float32), y), num_parallel_calls=AUTOTUNE)
    return ds


# Augmentation — horizontal_flip DISABLED (ASL is not mirror-symmetric).
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomRotation(0.08),
    tf.keras.layers.RandomTranslation(0.1, 0.1),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomBrightness(0.1),
    tf.keras.layers.RandomContrast(0.1),
], name='augmentation')


def augment(image, label):
    image = data_augmentation(image, training=True)
    return image, label


train_dir = os.path.join(DATA_DIR, "train")
val_dir   = os.path.join(DATA_DIR, "val")

train_ds = load_dataset(train_dir, shuffle=True)
val_ds   = load_dataset(val_dir,   shuffle=False)

num_classes = len(train_ds.class_names)
print(f"Classes      : {num_classes}")
print(f"Train batches: {len(train_ds)}  |  Val batches: {len(val_ds)}")

model = build_model(input_shape=(*IMG_SIZE, 3), num_classes=num_classes)

# Augment → prefetch (no cache on train to avoid RAM overflow on large datasets)
train_ds_aug = (train_ds
                .shuffle(2000)
                .map(augment, num_parallel_calls=AUTOTUNE)
                .prefetch(AUTOTUNE))

val_ds_cached = (val_ds
                 .cache()
                 .prefetch(AUTOTUNE))


# ---------------------------------------------------------------------------
# Phase 1: Train head only (MobileNetV2 base frozen)
# ---------------------------------------------------------------------------
os.makedirs("checkpoints", exist_ok=True)

callbacks_phase1 = [
    EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6, verbose=1),
    ModelCheckpoint("checkpoints/phase1_best.keras", monitor='val_accuracy', save_best_only=True, verbose=1),
]

print("\n===== Phase 1: Training head (MobileNetV2 frozen) =====")
model.fit(train_ds_aug, epochs=20, validation_data=val_ds_cached, callbacks=callbacks_phase1)

loss, acc = model.evaluate(val_ds_cached, verbose=0)
print(f"\nPhase 1 complete — val accuracy: {acc * 100:.2f}%")


# ---------------------------------------------------------------------------
# Phase 2: Fine-tune top 20 MobileNetV2 layers
# Only run this if Phase 1 val_acc > ~70% and you have enough RAM.
# ---------------------------------------------------------------------------
if acc >= 0.70:
    print("\n===== Phase 2: Fine-tuning top 20 MobileNetV2 layers =====")
    model = fine_tune_model(model, num_unfreeze=20)

    callbacks_phase2 = [
        EarlyStopping(monitor='val_accuracy', patience=6, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7, verbose=1),
        ModelCheckpoint("checkpoints/phase2_best.keras", monitor='val_accuracy', save_best_only=True, verbose=1),
    ]

    model.fit(train_ds_aug, epochs=25, validation_data=val_ds_cached, callbacks=callbacks_phase2)

    loss, acc = model.evaluate(val_ds_cached, verbose=0)
    print(f"\nPhase 2 complete — val accuracy: {acc * 100:.2f}%")
else:
    print(f"\nPhase 1 val_acc={acc*100:.2f}% < 70% — skipping fine-tuning.")
    print("Consider training more epochs or checking your data split.")


# ---------------------------------------------------------------------------
# Save weights for DFL node loading
# ---------------------------------------------------------------------------
weights_list = [w.tolist() for w in model.get_weights()]
with open("weights_image.json", "w") as f:
    json.dump(weights_list, f)

print("Saved weights_image.json")