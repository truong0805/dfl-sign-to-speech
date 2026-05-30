"""
train_image.py — standalone full-dataset trainer for MobileNetV2.

Run this ONCE on the full dataset (before DFL) to get an upper-bound benchmark.
The resulting val_acc is the ceiling your DFL nodes should approach after enough rounds.

CHANGES FROM PREVIOUS VERSION:
  - Input resolution updated to 160×160 (was 128×128) to match improved model.
  - Augmentation aligned with node.py (rotation 15%, translation 10%, zoom 15%,
    brightness 20%, contrast 20%). Previously the standalone trainer used different
    augmentation strengths than the nodes — this causes the benchmark to be
    unrepresentative of what the distributed system actually trains on.
  - Cosine LR schedule aligned with model_image.py (5e-4 initial, was 1e-3).
  - Phase 2 fine-tuning now unfreezes 30 layers (was 20) to match node.py.
  - Label smoothing (0.1) used in both phases, consistent with model_image.py.
"""

import os
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from model_image import build_model, fine_tune_model
import json

TRAIN_DIR  = os.environ.get("TRAIN_DIR", "resized_dataset")
VAL_DIR    = os.environ.get("VAL_DIR", "global_val")
AUTOTUNE   = tf.data.AUTOTUNE
IMG_SIZE   = (160, 160)    # Updated from 128×128
BATCH_SIZE = 32            # safe for standalone training on a 16 GB machine


def load_dataset(split_dir, shuffle):
    ds = tf.keras.utils.image_dataset_from_directory(
        split_dir,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='categorical',
        shuffle=shuffle,
    )
    # Cast to float32 in [0, 255].
    # The Rescaling layer inside build_model() converts to [-1, 1] at runtime.
    ds = ds.map(lambda x, y: (tf.cast(x, tf.float32), y), num_parallel_calls=AUTOTUNE)
    return ds


# Augmentation — kept in sync with node.py so this benchmark is representative.
# Horizontal flip DISABLED: ASL is NOT mirror-symmetric (e.g. J, Z differ from their mirrors).
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomRotation(0.15),
    tf.keras.layers.RandomTranslation(0.10, 0.10),
    tf.keras.layers.RandomZoom(0.15),
    tf.keras.layers.RandomBrightness(0.20),
    tf.keras.layers.RandomContrast(0.20),
], name='augmentation')


def augment(image, label):
    image = data_augmentation(image, training=True)
    return image, label


train_ds = load_dataset(TRAIN_DIR, shuffle=True)
val_ds   = load_dataset(VAL_DIR,   shuffle=False)

num_classes = len(train_ds.class_names)
print(f"Classes      : {num_classes}")
print(f"Train batches: {len(train_ds)}  |  Val batches: {len(val_ds)}")

model = build_model(input_shape=(*IMG_SIZE, 3), num_classes=num_classes)

train_ds_aug = (train_ds
                .shuffle(2000)
                .map(augment, num_parallel_calls=AUTOTUNE)
                .prefetch(AUTOTUNE))

val_ds_cached = (val_ds
                 .cache()
                 .prefetch(AUTOTUNE))


# ---------------------------------------------------------------------------
# Phase 1: Train classification head only (MobileNetV2 backbone frozen)
# ---------------------------------------------------------------------------
os.makedirs("checkpoints", exist_ok=True)

callbacks_phase1 = [
    EarlyStopping(monitor='val_accuracy', patience=5,
                  restore_best_weights=True, verbose=1),
    ModelCheckpoint("checkpoints/phase1_best.keras",
                    monitor='val_accuracy', save_best_only=True, verbose=1),
]

print("\n===== Phase 1: Training head (MobileNetV2 frozen) =====")
history1 = model.fit(train_ds_aug, epochs=20,
                     validation_data=val_ds_cached,
                     callbacks=callbacks_phase1)

loss, acc = model.evaluate(val_ds_cached, verbose=0)
print(f"\nPhase 1 complete — val accuracy: {acc * 100:.2f}%")

# Log train vs val gap so you can spot overfitting early
final_epoch = len(history1.history['accuracy']) - 1
train_acc_final = history1.history['accuracy'][final_epoch]
print(f"Phase 1 final train acc: {train_acc_final * 100:.2f}%  "
      f"val acc: {acc * 100:.2f}%  "
      f"gap: {(train_acc_final - acc) * 100:.2f}pp")
if (train_acc_final - acc) > 0.10:
    print("WARNING: train/val gap > 10pp — overfitting detected. "
          "Consider stronger augmentation or higher dropout.")


# ---------------------------------------------------------------------------
# Phase 2: Fine-tune top 30 MobileNetV2 layers
# ---------------------------------------------------------------------------
if acc >= 0.70:
    print("\n===== Phase 2: Fine-tuning top 30 MobileNetV2 layers =====")
    model = fine_tune_model(model, num_unfreeze=30)

    callbacks_phase2 = [
        EarlyStopping(monitor='val_accuracy', patience=6,
                      restore_best_weights=True, verbose=1),
        ModelCheckpoint("checkpoints/phase2_best.keras",
                        monitor='val_accuracy', save_best_only=True, verbose=1),
    ]

    history2 = model.fit(train_ds_aug, epochs=25,
                         validation_data=val_ds_cached,
                         callbacks=callbacks_phase2)

    loss, acc = model.evaluate(val_ds_cached, verbose=0)
    print(f"\nPhase 2 complete — val accuracy: {acc * 100:.2f}%")

    final_epoch2 = len(history2.history['accuracy']) - 1
    train_acc_final2 = history2.history['accuracy'][final_epoch2]
    print(f"Phase 2 final train acc: {train_acc_final2 * 100:.2f}%  "
          f"val acc: {acc * 100:.2f}%  "
          f"gap: {(train_acc_final2 - acc) * 100:.2f}pp")
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