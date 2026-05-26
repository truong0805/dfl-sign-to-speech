from PIL import Image
import os

def resize_dataset(input_dir, output_dir, size=(128, 128)):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for class_name in os.listdir(input_dir):
        class_path = os.path.join(input_dir, class_name)
        if not os.path.isdir(class_path):
            continue

        target_class_path = os.path.join(output_dir, class_name)
        os.makedirs(target_class_path, exist_ok=True)

        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)

            try:
                img = Image.open(img_path).convert("RGB")
                img = img.resize(size)  # resize here

                save_path = os.path.join(target_class_path, img_name)
                img.save(save_path)

            except Exception as e:
                print(f"Skipping {img_name}: {e}")

if __name__ == "__main__":
    INPUT_DIR = os.environ.get("ASL_DATASET_DIR", "asl_dataset")
    resize_dataset(INPUT_DIR, "resized_dataset", size=(128, 128))