import os
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from model_image import build_model, fine_tune_model
from utils import get_weights
import json

DATA_DIR  = os.environ.get("IMAGE_DATA_DIR", "resized_dataset")
AUTOTUNE  = tf.data.AUTOTUNE
IMG_SIZE   = (128, 128)
BATCH_SIZE = 64

# ---------------------------------------------------------------------------
# IMPORTANT — EfficientNetB0 has built-in preprocessing (include_preprocessing=True).
# Pass raw pixel values as float32 in [0, 255]. Do NOT divide by 255.
# ---------------------------------------------------------------------------

def load_dataset(split_dir, shuffle):
    ds = tf.keras.utils.image_dataset_from_directory(
        split_dir,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='categorical',
        shuffle=shuffle,
    )
    # Cast to float32; pixels stay in [0, 255] — EfficientNet preprocesses internally
    ds = ds.map(lambda x, y: (tf.cast(x, tf.float32), y), num_parallel_calls=AUTOTUNE)
    return ds


# ---------------------------------------------------------------------------
# Augmentation — horizontal_flip DISABLED (ASL signs are not mirror-symmetric)
# ---------------------------------------------------------------------------
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomRotation(0.08),
    tf.keras.layers.RandomTranslation(0.1, 0.1),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomBrightness(0.2),
    tf.keras.layers.RandomContrast(0.1),
], name='augmentation')

def augment(image, label):
    image = data_augmentation(image, training=True)
    return image, label


train_dir = os.path.join(DATA_DIR, "train")
val_dir   = os.path.join(DATA_DIR, "val")

# Load base datasets (no cache yet — augmentation must come before cache)
train_ds = load_dataset(train_dir, shuffle=True)
val_ds   = load_dataset(val_dir,   shuffle=False)

num_classes = len(train_ds.class_names)
print(f"Classes      : {num_classes}")
print(f"Train batches: {len(train_ds)}  |  Val batches: {len(val_ds)}")

# ---------------------------------------------------------------------------
# BUG FIX: Build the model exactly ONCE.
# Previous version called build_model() 3 times — twice as dead code and
# once actually used. The extra calls wasted RAM/time and risked OOM crashes.
# ---------------------------------------------------------------------------
model = build_model(num_classes=num_classes)

# ---------------------------------------------------------------------------
# BUG FIX: Apply augmentation THEN cache, not the other way around.
# If you cache before augmenting, every epoch sees the same fixed augmented
# images — the entire point of augmentation is lost.
# Correct order: shuffle → augment → cache → prefetch
# Val set: no augmentation, just cache for speed.
# ---------------------------------------------------------------------------
train_ds_aug = (train_ds
                .shuffle(2000)
                .map(augment, num_parallel_calls=AUTOTUNE)
                .cache()
                .prefetch(AUTOTUNE))

val_ds_cached = (val_ds
                 .cache()
                 .prefetch(AUTOTUNE))


# ---------------------------------------------------------------------------
# Phase 1: Train head only (EfficientNetB0 base frozen)
# ---------------------------------------------------------------------------
os.makedirs("checkpoints", exist_ok=True)

callbacks_phase1 = [
    EarlyStopping(monitor='val_accuracy', patience=4, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6, verbose=1),
    ModelCheckpoint("checkpoints/phase1_best.keras", monitor='val_accuracy', save_best_only=True, verbose=1),
]

print("\n===== Phase 1: Training head (EfficientNetB0 frozen) =====")
model.fit(train_ds_aug, epochs=15, validation_data=val_ds_cached, callbacks=callbacks_phase1)

loss, acc = model.evaluate(val_ds_cached, verbose=0)
print(f"\nPhase 1 complete — val accuracy: {acc * 100:.2f}%")


# ---------------------------------------------------------------------------
# Phase 2: Fine-tune top 30 EfficientNetB0 layers
# ---------------------------------------------------------------------------
print("\n===== Phase 2: Fine-tuning top 30 EfficientNetB0 layers =====")
model = fine_tune_model(model, num_unfreeze=30)

callbacks_phase2 = [
    EarlyStopping(monitor='val_accuracy', patience=6, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7, verbose=1),
    ModelCheckpoint("checkpoints/phase2_best.keras", monitor='val_accuracy', save_best_only=True, verbose=1),
]

model.fit(train_ds_aug, epochs=20, validation_data=val_ds_cached, callbacks=callbacks_phase2)

loss, acc = model.evaluate(val_ds_cached, verbose=0)
print(f"\nPhase 2 complete — val accuracy: {acc * 100:.2f}%")


# ---------------------------------------------------------------------------
# Save weights for DFL node loading
# ---------------------------------------------------------------------------
with open("weights_image.json", "w") as f:
    json.dump(get_weights(model), f)

print("Saved weights_image.json")