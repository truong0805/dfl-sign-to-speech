from PIL import Image
import os

def resize_dataset(input_dir, output_dir, size=(128, 128)):
    """
    Resize all images in input_dir to `size` and save to output_dir,
    preserving the class subfolder structure.

    FIX: Uses Image.LANCZOS resampling instead of the default nearest-neighbour.
    LANCZOS is a high-quality downsampling filter that preserves edge sharpness
    — important for hand gesture images where finger boundaries matter.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    classes = [
        d for d in os.listdir(input_dir)
        if os.path.isdir(os.path.join(input_dir, d))
    ]
    print(f"Found {len(classes)} class folders: {sorted(classes)}")

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

            # Skip non-files (e.g. nested folders)
            if not os.path.isfile(img_path):
                continue

            try:
                img = Image.open(img_path).convert("RGB")
                # LANCZOS = high-quality resampling, much better than default nearest-neighbour
                img = img.resize(size, Image.LANCZOS)

                save_path = os.path.join(target_class_path, img_name)
                img.save(save_path)
                saved += 1

            except Exception as e:
                print(f"  Skipping {img_name}: {e}")
                failed += 1

        print(f"  [{class_name}] {saved} saved, {failed} failed")
        total_saved  += saved
        total_failed += failed

    print(f"\nDone. Total saved: {total_saved} | Total failed: {total_failed}")


if __name__ == "__main__":
    # Update INPUT_DIR to match where your new dataset's train folder is located.
    # From your project structure this should be: data/asl_alphabet_train
    INPUT_DIR  = "data/asl_alphabet_train"
    OUTPUT_DIR = "resized_dataset"

    resize_dataset(INPUT_DIR, OUTPUT_DIR, size=(128, 128))