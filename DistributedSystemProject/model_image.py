from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
def build_model(input_shape=(128, 128, 3), num_classes=26):

    base_model = EfficientNetB0(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet',
    )

    # Freeze most layers
    base_model.trainable = True

    for layer in base_model.layers[:-20]:
        layer.trainable = False

    model = models.Sequential([

        # Normalize images
        layers.Rescaling(scale=1./255),

        base_model,

        layers.GlobalAveragePooling2D(),

        layers.BatchNormalization(),

        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),

        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),

        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),

        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=Adam(learning_rate=1e-4),

        loss=tf.keras.losses.CategoricalCrossentropy(
            label_smoothing=0.1
        ),

        metrics=['accuracy']
    )

    return model


def fine_tune_model(model, num_unfreeze=50):

    # Rescaling layer = index 0
    # EfficientNet = index 1
    base_model = model.layers[1]

    base_model.trainable = True

    for layer in base_model.layers[:-num_unfreeze]:
        layer.trainable = False

    # Keep BatchNorm frozen
    for layer in base_model.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False

    model.compile(

        optimizer=Adam(learning_rate=1e-5),

        loss=tf.keras.losses.CategoricalCrossentropy(
            label_smoothing=0.1
        ),

        metrics=['accuracy']
    )

    return model

# keep these unchanged from your existing model.py
def serialize_weights(model) -> bytes: ...
def deserialize_weights(model, data: bytes) -> None: ...
def get_weight_count(model) -> int: ...