import os
import zipfile
import argparse
from google.colab import drive
import shutil

def connect_to_drive():
    """
    Connects to Google Drive and mounts it.
    """
    print("Connecting to Google Drive...")
    drive.mount('/content/drive')
    print("Google Drive connected!")

def download_and_prepare(base_dir, dataset_name, project_dir):
    """
    Downloads and unzips dataset files from Google Drive to the local project directory.
    
    Args:
        base_dir (str): Base directory where datasets are stored in Google Drive.
        dataset_name (str): Name of the dataset (e.g., 'MSCOCO_dataset', 'Flickr30k_dataset').
        project_dir (str): Path to the local project directory where files will be saved.
    """
    # Define paths in Google Drive
    dataset_path = os.path.join(base_dir, dataset_name)
    data_path = os.path.join(dataset_path, "data")
    meta_data_path = os.path.join(dataset_path, "meta_data")

    # Check if the paths exist in Google Drive
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data path not found in Google Drive: {data_path}")
    if not os.path.exists(meta_data_path):
        raise FileNotFoundError(f"Meta-data path not found in Google Drive: {meta_data_path}")

    # Create local directories for data and meta-data
    local_data_dir = os.path.join(project_dir, "data")
    local_meta_data_dir = os.path.join(project_dir, "meta_data")
    os.makedirs(local_data_dir, exist_ok=True)
    os.makedirs(local_meta_data_dir, exist_ok=True)

    # Download and unzip files from the 'data' directory
    for file_name in os.listdir(data_path):
        file_path = os.path.join(data_path, file_name)
        local_file_path = os.path.join(local_data_dir, file_name)

        if file_name.endswith(".zip"):
            print(f"Downloading and unzipping: {file_name} ...")
            shutil.copy(file_path, local_file_path)  # Copy zip file locally
            with zipfile.ZipFile(local_file_path, 'r') as zip_ref:
                zip_ref.extractall(local_data_dir)  # Unzip into the local 'data' directory
            os.remove(local_file_path)  # Remove the zip file after extraction
            print(f"Unzipped to: {local_data_dir}")
        else:
            print(f"Skipping non-zip file: {file_name}")

    # Copy the JSON meta-data file to the local 'meta_data' directory
    for meta_file in os.listdir(meta_data_path):
        if meta_file.endswith(".json"):
            print(f"Downloading meta-data: {meta_file} ...")
            shutil.copy(os.path.join(meta_data_path, meta_file), local_meta_data_dir)

    print(f"All files for {dataset_name} have been downloaded and prepared in {project_dir}")



if __name__ == "__main__":
    # Connect to Google Drive
    connect_to_drive()

    # Argument parser
    parser = argparse.ArgumentParser(
        description="Download and prepare dataset files from Google Drive to the local project directory."
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default="/content/drive/MyDrive",
        help="Base directory in Google Drive where datasets are stored.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        choices=["MSCOCO_Dataset", "Flickr30k_Dataset"],
        help="Name of the dataset to prepare. Options: 'MSCOCO_dataset', 'Flickr30k_dataset'.",
    )
    parser.add_argument(
        "--project_dir",
        type=str,
        default=os.getcwd(),
        help="Local project directory where files will be saved.",
    )

    args = parser.parse_args()

    # Call the function with the provided arguments
    download_and_prepare(base_dir=args.base_dir, dataset_name=args.dataset_name, project_dir=args.project_dir)
