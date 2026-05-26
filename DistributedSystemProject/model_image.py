from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy


def build_model(input_shape=(128, 128, 3), num_classes=26):

    # Load pretrained backbone
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )

        # Freeze MOST layers
    base_model.trainable = True

    for layer in base_model.layers[:-30]:
        layer.trainable = False
    # OPTIONAL:
    # Fine-tune last few layers for better webcam adaptation
    for layer in base_model.layers[-20:]:
        layer.trainable = True

    model = models.Sequential([
        base_model,

        layers.GlobalAveragePooling2D(),

        layers.Dense(256, activation='relu'),

        # Stronger regularization
        layers.Dropout(0.5),

        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=Adam(learning_rate=1e-4),

        # LABEL SMOOTHING REDUCES OVERCONFIDENCE
        loss=CategoricalCrossentropy(
            label_smoothing=0.1
        ),

        metrics=['accuracy']
    )

    return model