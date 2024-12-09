import os 
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader 
from transformers import PreTrainedTokenizerBase
import torch


class ImageCaptionDataset:
    def __init__(self,image_caption_pairs,processor,transform=None):
        """
        Initialize the dataset.
        Args:
            image_caption_pairs (list): List of (image_id, caption_id, image_path, caption).
            transform (callable, optional): Optional transform to be applied on the image.
        """
        self.processor = processor
        self.image_caption_pairs = image_caption_pairs

    def __len__(self):
        return len(self.image_caption_pairs)
    
    def __getitem__(self, idx):
        """
        Fetch a single image-caption pair.
        Args:
            idx (int): Index of the image-caption pair.
        Returns:
            dict: {'image': image_tensor, 'caption': caption, image_id: image_id, caption_id: caption_id}
        """
        try:
            image_id, caption_id, image_path, caption = self.image_caption_pairs[idx]
            with Image.open(image_path) as image:
                if self.processor:
                    processed = self.processor(images=image, return_tensors="pt")
                    return {
                        "image_id": image_id,
                        "caption_id": caption_id,
                        "pixel_values": processed["pixel_values"].squeeze(0),  # Remove batch dim
                        "caption": caption,
                    }
        except FileNotFoundError:
            print(f"Image not found: {image_path}")
        except OSError as e:
            print(f"Corrupted image: {image_path}, Error: {e}")
        return None  # Return None if any error occurs

class CustomCollateFn:
    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        """
        Initialize the collate function with a tokenizer.
        Args:
            tokenizer: A tokenizer to handle text tokenization and padding.
        """
        self.tokenizer = tokenizer

    def __call__(self, batch):
        """
        Custom collate function to handle dynamic padding for captions.
        Args:
            batch: A list of items fetched by the dataset's __getitem__ method.
        Returns:
            dict: A dictionary with padded captions and images.
        """
        # Filter out None entries from __getitem__ (e.g., missing images)
        batch = [item for item in batch if item is not None]

        image_ids = [item["image_id"] for item in batch]
        caption_ids = [item["caption_id"] for item in batch]
        pixel_values = torch.stack([item["pixel_values"] for item in batch])
        captions = [item["caption"] for item in batch]

        # Tokenize and pad captions
        tokenized_captions = self.tokenizer(
            captions, padding=True, truncation=True, return_tensors="pt"
        )

        return {
            "image_ids": image_ids,
            "caption_ids": caption_ids,
            "pixel_values": pixel_values,
            "input_ids": tokenized_captions["input_ids"],
            "attention_mask": tokenized_captions["attention_mask"],
        }


def create_data_loader(dataset, batch_size=64, num_workers=2,collate_fn=None):
    """
    Create a DataLoader object for the image caption pairs.
    Args:
        image_caption_pairs (list): List of (image_id, caption_id, image_path, caption).
        batch_size (int, optional): Batch size for the data loader.
        num_workers (int, optional): Number of workers for the data loader.
    Returns:
        DataLoader: DataLoader object for the image caption pairs.
    """
    return DataLoader(dataset,
                      batch_size=batch_size,
                      num_workers=num_workers,
                      collate_fn=collate_fn)
        
