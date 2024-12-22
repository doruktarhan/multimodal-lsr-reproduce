import os
import argparse
from tqdm import tqdm
from loaders.dataset import MSCOCOdataset, Flickr30kdataset
from loaders.dataloader import ImageCaptionDataset, create_data_loader, CustomCollateFn
from models.blip import BLIP
from models.blip2 import BLIP2AvgPooling,BLIP2MaxPooling
from models.clip import CLIP
import pandas as pd

# Global variable for image folder
PATH_TO_MSCOCO_FOLDER = '/Users/doruktarhan/Desktop/MSCOCO_trial_images_small'
PATH_TO_FLICKR_FOLDER = '/home/scur2879/multimodal-lsr-reproduce/multimodal-lsr-reproduce/image_data/flickr30k-images'



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
        "id": image_ids,
        "emb": [emb.tolist() for emb in image_embeddings]
    }).drop_duplicates(subset="id", keep="first")
    img_parquet_path = os.path.join(save_dir, "img_embs.parquet")
    img_df.to_parquet(img_parquet_path, engine="pyarrow")
    print(f"Saved image embeddings to {img_parquet_path}")

    # Create a DataFrame for text embeddings
    text_df = pd.DataFrame({
        "id": caption_ids,
        "emb": [emb.tolist() for emb in image_embeddings]
    })
    text_parquet_path = os.path.join(save_dir, "text_embs.parquet")
    text_df.to_parquet(text_parquet_path, engine="pyarrow",index = False)
    print(f"Saved text embeddings to {text_parquet_path}")



def main(args):

    # Dataset and metadata initialization
    if args.dataset_name == "mscoco":
        #download and prepare the dataset
        meta_data_path = 'meta_data_old_without_filtering/dataset_coco.json' # Path to the MSCOCO dataset
        dataset = MSCOCOdataset(meta_data_path,PATH_TO_MSCOCO_FOLDER)
        image_caption_pairs = dataset.get_image_caption_pairs()
    elif args.dataset_name == "flickr30k":
        meta_data_path = 'meta_data_old_without_filtering/dataset_flickr30k.json' # Path to the MSCOCO dataset
        dataset = Flickr30kdataset(meta_data_path,PATH_TO_FLICKR_FOLDER)
        image_caption_pairs = dataset.get_image_caption_pairs()
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset_name}")

    # Model initialization
    if args.model_name == "blip":
        model = BLIP()
        processor = model.processor
    elif args.model_name == "blip2avg":
        model = BLIP2AvgPooling()
        processor = model.processor
    elif args.model_name == "blip2max":
        model = BLIP2MaxPooling()
        processor = model.processor
    elif args.model_name == "clip":
        model = CLIP()
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
        

    save_dir = os.path.join(args.save_dir,f'{args.dataset_name}-{args.model_name}-dense')
    os.makedirs(save_dir,exist_ok=True)
    # Save embeddings to Parquet files
    save_embeddings_to_parquet(all_image_embeddings, all_text_embeddings, all_image_ids, all_caption_ids,save_dir )




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
        default="mscoco",
        help="Name of the dataset to use. Options: 'mscoco'.",
    )
    
    # Model argument
    parser.add_argument(
        "--model_name",
        type=str,
        choices=["blip","blip2avg","blip2max","clip"],  # Add more model options as you support them
        default="clip",
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

