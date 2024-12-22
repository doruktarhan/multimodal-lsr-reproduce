from loaders.dataset import MSCOCOdataset
from loaders.dataloader import ImageCaptionDataset, create_data_loader, CustomCollateFn
from tqdm import tqdm
import torch
from models.blip import BLIP
import os



#testing the dataset.py
meta_data_path = 'data/dataset_coco.json'

coco= MSCOCOdataset(meta_data_path)
image_caption_pairs = coco.get_image_caption_pairs()




#initialize the model
model = BLIP()
processor = model.processor

#initialize the dataset
coco_data = ImageCaptionDataset(image_caption_pairs,processor=processor)
batch_size = 16
tokenizer = processor.tokenizer
collate_fn = CustomCollateFn(tokenizer = tokenizer)



#load the dataloader from dataloader.py
data_loader = create_data_loader(coco_data,
                                  batch_size=batch_size,
                                  collate_fn=collate_fn
                                  )

# Directory to save test embeddings
save_path = f"test_embeddings/{model.name}_test"
os.makedirs(save_path, exist_ok=True)  # Create the directory if it doesn't exist



for batch in tqdm(data_loader, desc="Processing batches"):
    print(batch.keys())
    print("Batch pixel values shape:", batch["pixel_values"].shape)
    print("Batch input IDs shape:", batch["input_ids"].shape)
    images =  batch["pixel_values"]
    input_ids = batch["input_ids"]
    image_ids = batch['image_ids']
    caption_ids = batch['caption_ids']
    attention_mask = batch['attention_mask']

    outputs = model.forward(batch)

    # Save the outputs
    model.save_embeddings(
        outputs=outputs,
        image_ids=batch["image_ids"],
        caption_ids=batch["caption_ids"],
        save_path=os.path.join(save_path, "test_batch.pt"),
    )

    print(f"Embeddings saved to {os.path.join(save_path, 'test_batch.pt')}")
    break
