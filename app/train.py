from tensorflow.keras.preprocessing.image import ImageDataGenerator
from model_image import build_model
from utils import get_weights

def train_local_model():
    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )

    train_data = datagen.flow_from_directory(
        '/app/data_shards',   # 🔥 KHÔNG có /train
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical',
    )

    model = build_model(num_classes=train_data.num_classes)
    model.fit(train_data, epochs=5)
    return model.get_weights()