from PIL import Image
import os
import shutil


def pad_and_resize(img: Image.Image, size=(160, 160)) -> Image.Image:
    """
    Pad image to square with black letterbox borders, then resize.

    Why this matters:
        The ASL training images are NOT square — each sign has a different
        natural aspect ratio (e.g. 'H' is very wide at ~2:1, 'B' is tall
        at ~1:2). A naive img.resize(160, 160) squashes/stretches each class
        differently, baking a unique per-class distortion into the training data.
        The model then learns to use that distortion as a class cue, which means
        it fails on real camera images (which always show natural proportions).

        This function pads the shorter dimension with black pixels first so the
        hand shape is preserved, matching what the model will see at inference.
    """
    w, h = img.size
    max_dim = max(w, h)
    # Create a square black canvas and paste the original image centered
    square = Image.new("RGB", (max_dim, max_dim), (0, 0, 0))
    paste_x = (max_dim - w) // 2
    paste_y = (max_dim - h) // 2
    square.paste(img, (paste_x, paste_y))
    return square.resize(size, Image.LANCZOS)


def resize_dataset_folder(input_dir, output_dir, size=(160, 160)):
    """
    Resize all images in input_dir to `size` and save to output_dir,
    preserving the class subfolder structure.

    Uses pad_and_resize() instead of a direct resize to preserve hand
    aspect ratios across all ASL sign classes.
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
                img = pad_and_resize(img, size)  # aspect-ratio-safe resize

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