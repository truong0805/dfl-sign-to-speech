import os
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from model_image import build_model, fine_tune_model
from utils import get_weights
import json

DATA_DIR  = os.environ.get("IMAGE_DATA_DIR", "resized_dataset")
AUTOTUNE  = tf.data.AUTOTUNE

# ---------------------------------------------------------------------------
# IMPORTANT — EfficientNetB3 has a Rescaling layer that normalizes [0-255] → [0, 1].
# This means the model expects raw pixel values as float32 in [0, 255].
# Do NOT rescale to [0, 1] before feeding into the model.
# The model's Rescaling layer handles normalization internally.
# ---------------------------------------------------------------------------

IMG_SIZE   = (128, 128)
BATCH_SIZE = 8  # Reduced to 8 for minimal RAM usage (training will take longer)

def load_dataset(split_dir, shuffle):
    """Load images and cast to float32 [0, 255] - streaming mode to avoid caching"""
    ds = tf.keras.utils.image_dataset_from_directory(
        split_dir,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='categorical',
        shuffle=shuffle,
    )
    # Cast to float32 (pixels stay in [0, 255] — EfficientNet preprocesses internally)
    ds = ds.map(lambda x, y: (tf.cast(x, tf.float32), y), num_parallel_calls=AUTOTUNE)
    if shuffle:
        # Removed .cache() to prevent loading entire dataset into RAM
        # Data loads from disk on-the-fly (slower but uses ~80% less memory)
        ds = ds.shuffle(1000, reshuffle_each_iteration=True).prefetch(tf.data.AUTOTUNE)
    else:
        # No caching, stream directly from disk
        ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


# ---------------------------------------------------------------------------
# In-graph augmentation (GPU-accelerated, applied only during training)
# horizontal_flip is DISABLED — ASL signs are NOT mirror-symmetric.
# ---------------------------------------------------------------------------
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomRotation(0.08),
    tf.keras.layers.RandomTranslation(0.1, 0.1),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomBrightness(0.2),
    tf.keras.layers.RandomContrast(0.1),
], name='augmentation')

# Load datasets
train_dir = os.path.join(DATA_DIR, "train")
val_dir   = os.path.join(DATA_DIR, "val")

train_ds = load_dataset(train_dir, shuffle=True)
val_ds   = load_dataset(val_dir,   shuffle=False)

num_classes = len(train_ds.class_names)
print(f"Classes: {num_classes}")
print(f"Train batches: {len(train_ds)}  |  Val batches: {len(val_ds)}")

# Apply augmentation to training set
def augment(image, label):
    image = data_augmentation(image, training=True)
    return image, label

train_ds_aug = train_ds.map(augment, num_parallel_calls=AUTOTUNE)

# Create model
model = build_model(num_classes=num_classes)

# Phase 1: Train head only
print("\n===== Phase 1: Training head (EfficientNetB0 frozen) =====")

callbacks_phase1 = [
    EarlyStopping(
        monitor='val_accuracy',
        patience=4,
        restore_best_weights=True,
        verbose=1,
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        verbose=1,
    ),
]

os.makedirs("checkpoints", exist_ok=True)

model.fit(
    train_ds_aug,
    epochs=15,
    validation_data=val_ds,
    callbacks=callbacks_phase1,
)

loss, acc = model.evaluate(val_ds, verbose=0)
print(f"\nPhase 1 complete — val accuracy: {acc * 100:.2f}%")

# Phase 2: Fine-tune
print("\n===== Phase 2: Fine-tuning top 30 EfficientNetB0 layers =====")

model = fine_tune_model(model, num_unfreeze=30)

callbacks_phase2 = [
    EarlyStopping(
        monitor='val_accuracy',
        patience=6,
        restore_best_weights=True,
        verbose=1,
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1,
    ),
]

model.fit(
    train_ds_aug,
    epochs=20,
    validation_data=val_ds,
    callbacks=callbacks_phase2,
)

loss, acc = model.evaluate(val_ds, verbose=0)
print(f"\nPhase 2 complete — val accuracy: {acc * 100:.2f}%")

# Save final model and weights
model.save("sign_language_model.keras")
with open("weights_image.json", "w") as f:
    json.dump(get_weights(model), f)

print("Model and weights saved successfully!")
