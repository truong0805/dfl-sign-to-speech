import pandas as pd
import numpy as np
from tensorflow.keras.utils import to_categorical
from model_mnist import build_model
from utils import get_weights
import json

# Load data
train = pd.read_csv("data/mnist/sign_mnist_train.csv")
test = pd.read_csv("data/mnist/sign_mnist_test.csv")

# Split X, y
X_train = train.iloc[:, 1:].values
y_train = train.iloc[:, 0].values

X_test = test.iloc[:, 1:].values
y_test = test.iloc[:, 0].values

# Reshape
X_train = X_train.reshape(-1, 28, 28, 1) / 255.0
X_test = X_test.reshape(-1, 28, 28, 1) / 255.0

# One-hot
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Build model
model = build_model(num_classes=y_train.shape[1])

# Train
model.fit(X_train, y_train, epochs=5)

# Evaluate
loss, acc = model.evaluate(X_test, y_test)
print("MNIST Accuracy:", acc)

# Save weights
with open("weights_mnist.json", "w") as f:
    json.dump(get_weights(model), f)