import numpy as np
from tensorflow.keras import layers, models
from utils import get_weights

def build_model(input_shape=(64, 64, 3), num_classes=36):
    model = models.Sequential([
        layers.Input(shape=input_shape),

        layers.Conv2D(32, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),

        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model
def get_weights_as_numpy(model):
    return model.get_weights()