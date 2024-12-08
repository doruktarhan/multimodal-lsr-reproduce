import os
import argparse
from tqdm import tqdm
from loaders.dataset import MSCOCOdataset, Flickr30kdataset
from loaders.dataloader import ImageCaptionDataset, create_data_loader, CustomCollateFn
from models.blip import BLIP
import pandas as pd
import shutil
import zipfile
from google.colab import drive


def save_embeddings_to_parquet(image_embeddings, text_embeddings, image_ids, caption_ids, save_dir):
    """
    Save the embeddings to Parquet files.

    Args:
        image_embeddings (list): List of image embeddings.
        text_embeddings (list): List of text embeddings.
        image_ids (list): List of image IDs.
        caption_ids (list): List of caption IDs.
        save_dir (str): Directory to save the Parquet files.
    """
    # Create a DataFrame for image embeddings
    img_df = pd.DataFrame({
        "index": image_ids,
        "embedding": [emb.tolist() for emb in image_embeddings]
    })
    img_parquet_path = os.path.join(save_dir, "img_embs.parquet")
    img_df.to_parquet(img_parquet_path, engine="pyarrow")
    print(f"Saved image embeddings to {img_parquet_path}")

    # Create a DataFrame for text embeddings
    text_df = pd.DataFrame({
        "index": caption_ids,
        "embedding": [emb.tolist() for emb in text_embeddings]
    })
    text_parquet_path = os.path.join(save_dir, "text_embs.parquet")
    text_df.to_parquet(text_parquet_path, engine="pyarrow")
    print(f"Saved text embeddings to {text_parquet_path}")

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



def main(args):
    #connect to drive
    connect_to_drive()


    # Dataset and metadata initialization
    if args.dataset_name == "mscoco":
        #download and prepare the dataset
        download_and_prepare('/content/drive/MyDrive', 'MSCOCO_Dataset',os.getcwd())
        meta_data_path = 'meta_data/dataset_coco.json' # Path to the MSCOCO dataset
        dataset = MSCOCOdataset(meta_data_path)
        image_caption_pairs = dataset.get_image_caption_pairs()
    elif args.dataset_name == "flickr30k":
        #download and prepare the dataset
        download_and_prepare('/content/drive/MyDrive', 'Flickr30k_Dataset',os.getcwd())
        meta_data_path = 'meta_data/dataset_flickr30k.json' # Path to the MSCOCO dataset
        dataset = Flickr30kdataset(meta_data_path)
        image_caption_pairs = dataset.get_image_caption_pairs()
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset_name}")

    # Model initialization
    if args.model_name == "blip":
        model = BLIP()
        processor = model.processor
    else:
        raise ValueError(f"Unsupported model: {args.model_name}")

    # Dataset and DataLoader
    data = ImageCaptionDataset(image_caption_pairs, processor=processor)
    collate_fn = CustomCollateFn(tokenizer=processor.tokenizer)
    data_loader = create_data_loader(data, batch_size=args.batch_size, collate_fn=collate_fn)


    # Create directory for saving embeddings
    os.makedirs(args.save_dir, exist_ok=True)

    # Data structures to store embeddings and IDs
    all_image_embeddings = []
    all_text_embeddings = []
    all_image_ids = []
    all_caption_ids = []
    # Process batches
    for batch in tqdm(data_loader, desc="Processing batches"):
        outputs = model.forward(batch)
        # Append embeddings and IDs to lists
        all_image_embeddings.extend(outputs["image_embeds"].cpu())
        all_text_embeddings.extend(outputs["text_embeds"].cpu())
        all_image_ids.extend(batch["image_ids"])
        all_caption_ids.extend(batch["caption_ids"])
        

    # Save embeddings to Parquet files
    save_embeddings_to_parquet(all_image_embeddings, all_text_embeddings, all_image_ids, all_caption_ids, args.save_dir)




if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser(
        description="Extract embeddings using a specified model and dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Dataset argument
    parser.add_argument(
        "--dataset_name",
        type=str,
        choices=["mscoco","flickr30k"],  # Add more dataset options as you support them
        default="flickr30k",
        help="Name of the dataset to use. Options: 'mscoco'.",
    )
    
    # Model argument
    parser.add_argument(
        "--model_name",
        type=str,
        choices=["blip"],  # Add more model options as you support them
        default="blip",
        help="Name of the model to use for embedding extraction. Options: 'blip'.",
    )
    
    # Batch size argument
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size to use for DataLoader. Note: Batch size might not impact inference much, as it depends on GPU/CPU memory availability.",
    )
    
    # Save directory argument
    parser.add_argument(
        "--save_dir",
        type=str,
        default="embeddings",
        help="Directory where embeddings will be saved.",
    )

    # Parsing arguments
    args = parser.parse_args()

    main(args)

