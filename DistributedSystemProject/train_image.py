from tensorflow.keras.preprocessing.image import ImageDataGenerator
from model_image import build_model
from utils import get_weights
import json

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_data = datagen.flow_from_directory(
    'D:/Python/data/image_dataset',   # 🔥 KHÔNG có /train
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

test_data = datagen.flow_from_directory(
    'D:/Python/data/image_dataset',
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

model = build_model(num_classes=train_data.num_classes)

model.fit(train_data, epochs=5)

loss, acc = model.evaluate(test_data)
print("Accuracy:", acc)

with open("weights_image.json", "w") as f:
    json.dump(get_weights(model), f)