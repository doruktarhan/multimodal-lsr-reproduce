from transformers import AutoProcessor, Blip2ForImageTextRetrieval
import torch
from torch.nn.functional import normalize,cosine_similarity


MODEL_NAME = "Salesforce/blip2-itm-vit-g"

class BLIP2AvgPooling:
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
        self.model = Blip2ForImageTextRetrieval.from_pretrained(MODEL_NAME).to(self.device)
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
            outputs = self.model(input_ids = input_ids,
                                    attention_mask = attention_mask,
                                    pixel_values = images, 
                                    use_image_text_matching_head = False,
                                    return_dict=True)
            # Average pooling over 32 image embeddings and normalize
            image_embeds = normalize(outputs["image_embeds"].mean(dim=1), p=2, dim=-1)
            text_embeds = normalize(outputs["text_embeds"], p=2, dim=-1)

        return {'image_embeds': image_embeds, 'text_embeds': text_embeds}

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




class BLIP2MaxPooling:
    def __init__(self, model=None, processor=None, device="cuda"):
        """
        Initialize the BLIP2 model.
        """
        self.name = "blip2max"
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        )
        self.model = Blip2ForImageTextRetrieval.from_pretrained(MODEL_NAME).to(self.device)
        self.processor = AutoProcessor.from_pretrained(MODEL_NAME)

    def forward(self, batch):
        """
        Forward pass of the model for batch processing.
        """
        images = batch["pixel_values"].to(self.device)
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)

        # Forward pass without training
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=images,
                use_image_text_matching_head=False,
                return_dict=True,
            )
            # Compute cosine similarity for each of the 32 image embeddings
            similarities = cosine_similarity(
                outputs["image_embeds"],  # Shape: [Batch Size, 32, 256]
                outputs["text_embeds"].unsqueeze(1),  # Shape: [Batch Size, 1, 256]
                dim=-1,
            )  # Shape: [Batch Size, 32]
            max_similarities, max_indices = similarities.max(dim=1)  # Shape: [Batch Size], [Batch Size]

            # Select the most similar image embedding for each text
            batch_indices = torch.arange(outputs["image_embeds"].size(0), device=self.device)
            selected_image_embeds = outputs["image_embeds"][batch_indices, max_indices, :]  # Shape: [Batch Size, 256]
            
            
            # Normalize embeddings for alignment
            selected_image_embeds = normalize(selected_image_embeds, p=2, dim=-1)
            text_embeds = normalize(outputs["text_embeds"], p=2, dim=-1)

        return {"image_embeds": selected_image_embeds, "text_embeds": text_embeds}
