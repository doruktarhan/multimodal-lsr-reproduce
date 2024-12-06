from transformers import AutoProcessor, BlipForImageTextRetrieval
import torch
from torch.nn.functional import normalize

MODEL_NAME = "Salesforce/blip-itm-base-coco"

class BLIP:
    def __init__(self, model = None, processor = None, device = "cuda"):
        """
        Initialize the BLIP model.
        Args:
            model_name (str): The name of the model to be used.
        """
        self.name = 'blip'
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = BlipForImageTextRetrieval.from_pretrained(MODEL_NAME)
        self.processor = AutoProcessor.from_pretrained(MODEL_NAME)

    def forward(self, batch):
        """
        Forward pass of the model.
        Args:
            batch (dict): The batch of data to be passed through the model.
        Returns:
            dict: The output of the model.
        """
        images = batch["pixel_values"].to(self.device)
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)

        # Forward pas withput training
        with torch.no_grad():
            with torch.no_grad():
                outputs = self.model(input_ids = input_ids,
                                     attention_mask = attention_mask,
                                     pixel_values = images, 
                                     use_itm_head = False,
                                     return_dict=True)
                proj_text_embedding = normalize(self.model.text_proj(outputs.question_embeds[:,0,:]))
                proj_image_embedding = normalize(self.model.vision_proj(outputs.last_hidden_state[:,0,:]))

        return {'image_embeds': proj_image_embedding, 'text_embeds': proj_text_embedding}

    def save_embeddings(self, outputs: dict, image_ids: list, caption_ids:list, save_path: str ):
        """
        Save the outputs to a file with image and caption IDs.
        Args:
            outputs (dict): Model outputs containing embeddings.
            image_ids (list): IDs for the images in the batch.
            caption_ids (list): IDs for the captions in the batch.
            save_path (str): Path to save the embeddings.
        """
        embeddings = {
            "image_embeds": outputs["image_embeds"].cpu(),
            "text_embeds": outputs["text_embeds"].cpu(),
            "image_ids": image_ids,
            "caption_ids": caption_ids,
        }
        torch.save(embeddings, save_path)