from transformers import CLIPProcessor, CLIPModel
import torch
from torch.nn.functional import normalize,cosine_similarity



MODEL_NAME = "openai/clip-vit-base-patch32"

class CLIP:
    def __init__(self, model = None, processor = None, device = "cuda"):
        """
        Initialize the BLIP2 model.
        Args:
            model_name (str): The name of the model to be used.
        """
        self.name = 'blip2avg'
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        )
        self.model = CLIPModel.from_pretrained(MODEL_NAME).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(MODEL_NAME)

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
            outputs = self.model(input_ids = input_ids,
                                    attention_mask = attention_mask,
                                    pixel_values = images,
                                    return_dict=True)
            # Average pooling over 32 image embeddings and normalize
            image_embeds = outputs["image_embeds"]
            text_embeds =outputs["text_embeds"]
            #debug check
            print(f"image embeds size: {image_embeds.shape}")
            print(f"text embeds size: {text_embeds.shape}")

        return {'image_embeds': image_embeds, 'text_embeds': text_embeds}