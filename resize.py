from PIL import Image
import os
import shutil

def resize_dataset_folder(input_dir, output_dir, size=(160, 160)):
    """
    Resize all images in input_dir to `size` and save to output_dir,
    preserving the class subfolder structure.
    """
    if os.path.exists(output_dir):
        print(f"Wiping existing directory: {output_dir}")
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    classes = [
        d for d in os.listdir(input_dir)
        if os.path.isdir(os.path.join(input_dir, d))
    ]
    print(f"Found {len(classes)} class folders in '{input_dir}': {sorted(classes)}")

    total_saved  = 0
    total_failed = 0

    for class_name in sorted(classes):
        class_path = os.path.join(input_dir, class_name)
        target_class_path = os.path.join(output_dir, class_name)
        os.makedirs(target_class_path, exist_ok=True)

        saved  = 0
        failed = 0

        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)

            if not os.path.isfile(img_path):
                continue

            try:
                img = Image.open(img_path).convert("RGB")
                img = img.resize(size, Image.LANCZOS)

                save_path = os.path.join(target_class_path, img_name)
                img.save(save_path)
                saved += 1

            except Exception as e:
                print(f"  Skipping {img_name} in {class_name}: {e}")
                failed += 1

        print(f"  [{class_name}] {saved} saved, {failed} failed")
        total_saved  += saved
        total_failed += failed

    print(f"Finished directory '{input_dir}' -> '{output_dir}'. Total saved: {total_saved} | Total failed: {total_failed}\n")


if __name__ == "__main__":
    TRAIN_INPUT = "ASL_Processed_Images/asl_processed/train"
    TRAIN_OUTPUT = "resized_dataset"
    
    TEST_INPUT = "ASL_Processed_Images/asl_processed/test"
    TEST_OUTPUT = "global_val"

    print("=== Processing Training Dataset ===")
    resize_dataset_folder(TRAIN_INPUT, TRAIN_OUTPUT, size=(160, 160))

    print("=== Processing Test/Global Validation Dataset ===")
    resize_dataset_folder(TEST_INPUT, TEST_OUTPUT, size=(160, 160))