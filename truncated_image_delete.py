from PIL import Image
import os
from tqdm import tqdm

def remove_corrupted_images(root_path):
    """
    Go through all folders in the given path, check for corrupted images, and remove them.
    
    Args:
        root_path (str): Path to the root directory to check for corrupted images.
    """
    corrupted_images = []

    # Get a list of all image files in the directory tree for tqdm
    all_files = []
    for root, _, files in os.walk(root_path):
        for file in files:
            if not file.endswith('.zip'):  # Skip zip files
                all_files.append(os.path.join(root, file))
    print('Checking for file {file}')
    # Iterate through all files with tqdm progress bar
    for img_path in tqdm(all_files, desc="Checking for corrupted images"):
        try:
            with Image.open(img_path) as img:
                img.verify()  # Check if the file is a valid image
        except (OSError, FileNotFoundError) as e:
            print(f"Corrupted image: {img_path}, Error: {e}")
            corrupted_images.append(img_path)

    # Remove corrupted images
    for corrupted_image in corrupted_images:
        try:
            os.remove(corrupted_image)
            print(f"Removed: {corrupted_image}")
        except FileNotFoundError:
            print(f"File already removed: {corrupted_image}")
        except Exception as e:
            print(f"Failed to remove {corrupted_image}, Error: {e}")

    print(f"Total corrupted images removed: {len(corrupted_images)}")