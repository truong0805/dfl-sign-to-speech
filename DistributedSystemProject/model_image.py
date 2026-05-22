from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam

def build_model(input_shape=(128, 128, 3), num_classes=36):
    # Load MobileNetV2 pretrained on ImageNet, without the top classifier
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    # Freeze the base — we only train our custom head
    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=Adam(learning_rate=1e-4),  # lower LR = more stable
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model