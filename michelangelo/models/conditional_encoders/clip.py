# -*- coding: utf-8 -*-

import torch
import numpy as np
from PIL import Image
from dataclasses import dataclass
from torchvision.transforms import Normalize
from transformers import CLIPModel, CLIPTokenizer
from transformers.utils import ModelOutput
from typing import Iterable, Optional, Union, List, Tuple


ImageType = Union[np.ndarray, torch.Tensor, Image.Image]


@dataclass
# This decorator is used to define a data class, which is a special type of class that is designed to hold data without adding functionality.
class CLIPEmbedOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    pooler_output: torch.FloatTensor = None
    embeds: torch.FloatTensor = None


class CLIPEncoder(torch.nn.Module):
    """
    This class defines a CLIPEncoder model, which is a PyTorch module that encapsulates the functionality of encoding images and texts 
    using the CLIP model.
    """

    # openai/clip-vit-base-patch32
    def __init__(self, model_path="./pretrained/clip-vit-base-patch32"):
        """
        Initializes the CLIPEncoder model with the specified model path.

        Args:
            model_path (str, optional): The path to the pre-trained CLIP model. Defaults to "./pretrained/clip-vit-base-patch32".
        """
        super().__init__()

        # Load the CLIP model and processor
        self.model: CLIPModel = CLIPModel.from_pretrained(model_path)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_path)
        self.image_preprocess = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        # Set the model to evaluation mode and freeze its parameters
        self.model.training = False
        for p in self.model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def encode_image(self, images: Iterable[Optional[ImageType]]):
        """
        Encodes a batch of images using the CLIP model.

        Args:
            images (Iterable[Optional[ImageType]]): A batch of images to be encoded. Each image can be a numpy array, torch tensor, or PIL image.

        Returns:
            CLIPEmbedOutput: The encoded image embeddings.
        """
        pixel_values = self.image_preprocess(images)

        vision_outputs = self.model.vision_model(pixel_values=pixel_values)

        pooler_output = vision_outputs[1]  # pooled_output
        image_features = self.model.visual_projection(pooler_output)

        visual_embeds = CLIPEmbedOutput(
            last_hidden_state=vision_outputs.last_hidden_state,
            pooler_output=pooler_output,
            embeds=image_features
        )

        return visual_embeds

    @torch.no_grad()
    def encode_text(self, texts: List[str]):
        """
        Encodes a batch of texts using the CLIP model.

        Args:
            texts (List[str]): A list of texts to be encoded.

        Returns:
            CLIPEmbedOutput: The encoded text embeddings.
        """
        text_inputs = self.tokenizer(texts, padding=True, return_tensors="pt")

        text_outputs = self.model.text_model(input_ids=text_inputs)

        pooler_output = text_outputs[1]  # pooled_output
        text_features = self.model.text_projection(pooler_output)

        text_embeds = CLIPEmbedOutput(
            last_hidden_state=text_outputs.last_hidden_state,
            pooler_output=pooler_output,
            embeds=text_features
        )

        return text_embeds

    def forward(self,
                images: Iterable[Optional[ImageType]],
                texts: List[str]) -> Tuple[CLIPEmbedOutput, CLIPEmbedOutput]:
        """
        Encodes both images and texts using the CLIP model and returns their embeddings.

        Args:
            images (Iterable[Optional[ImageType]]): A batch of images to be encoded.
            texts (List[str]): A list of texts to be encoded.

        Returns:
            Tuple[CLIPEmbedOutput, CLIPEmbedOutput]: A tuple containing the encoded image embeddings and the encoded text embeddings.
        """
        visual_embeds = self.encode_image(images)
        text_embeds = self.encode_text(texts)

        return visual_embeds, text_embeds







