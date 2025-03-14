# -*- coding: utf-8 -*-

import torch
from torch import nn
from einops import rearrange
from transformers import CLIPModel

from michelangelo.models.tsal.tsal_base import AlignedShapeAsLatentModule

class CLIPAlignedShapeAsLatentModule(AlignedShapeAsLatentModule):
    """
    This class extends the AlignedShapeAsLatentModule to integrate CLIP model for text and image embeddings.
    It is designed to handle the encoding of shape, image, and text inputs into a common latent space.
    """

    # openai/clip-vit-large-patch14
    def __init__(self, *,
                 shape_model,
                 clip_model_version: str = "./pretrained/clip-vit-large-patch14"):
        """
        Initializes the CLIPAlignedShapeAsLatentModule with a shape model and a CLIP model version.

        Args:
            shape_model: The shape model to be used for encoding shape inputs.
            clip_model_version (str, optional): The version of the CLIP model to use. Defaults to "./pretrained/clip-vit-large-patch14".
        """
        super().__init__()

        self.clip_model: CLIPModel = CLIPModel.from_pretrained(clip_model_version)
        # Freeze the CLIP model parameters to prevent them from being updated during training
        for params in self.clip_model.parameters():
            params.requires_grad = False

        self.shape_model = shape_model
        # Initialize the shape projection matrix with a normal distribution
        # nn.Parameter() mark the parameters as learnable parameters
        # torch.empty() generate the random tensor
        self.shape_projection = nn.Parameter(torch.empty(self.shape_model.width, self.clip_model.projection_dim))
        # initialize the parameters as N(0,std) to avoid gradient explosion
        nn.init.normal_(self.shape_projection, std=self.clip_model.projection_dim ** -0.5)

    def set_shape_model_only(self):
        """
        Sets the CLIP model to None, effectively disabling it.
        """
        self.clip_model = None

    def encode_shape_embed(self, surface, return_latents: bool = False):
        """
        Encodes the shape input into a latent space using the shape model and projects it into the CLIP model's space.

        Args:
            surface (torch.FloatTensor): The shape input tensor of shape [bs, n, 3 + c], where bs is the batch size, n is the number of points, 3 is the dimensionality of the points, and c is the number of additional features.
            return_latents (bool, optional): Whether to return the shape latents in addition to the projected embedding. Defaults to False.

        Returns:
            x (torch.FloatTensor): The projected shape embedding of shape [bs, projection_dim].
            shape_latents (torch.FloatTensor, optional): The shape latents of shape [bs, m, d], where m is the number of latents and d is the dimensionality of each latent.
        """
        pc = surface[..., 0:3]  # Extract the point cloud from the surface
        feats = surface[..., 3:]  # Extract normal vector from the surface

        shape_embed, shape_latents = self.shape_model.encode_latents(pc, feats)  # Encode the shape into latents
        x = shape_embed @ self.shape_projection  # Project the shape embedding into the CLIP model's space

        if return_latents:
            return x, shape_latents
        else:
            return x

    def encode_image_embed(self, image):
        """
        Encodes the image input into a latent space using the CLIP model.

        Args:
            image (torch.FloatTensor): The image input tensor of shape [bs, 3, h, w], where bs is the batch size, 3 is the number of color channels, h is the height, and w is the width.

        Returns:
            x (torch.FloatTensor): The image embedding of shape [bs, projection_dim].
        """
        x = self.clip_model.get_image_features(image)  # Get the image features from the CLIP model

        return x

    def encode_text_embed(self, text):
        """
        Encodes the text input into a latent space using the CLIP model.

        Args:
            text (torch.LongTensor): The text input tensor of shape [bs, num_templates, 77], where bs is the batch size, num_templates is the number of templates, and 77 is the sequence length.

        Returns:
            x (torch.FloatTensor): The text embedding of shape [bs, projection_dim].
        """
        x = self.clip_model.get_text_features(text)  # Get the text features from the CLIP model

        return x

    def forward(self, surface, image, text):
        """
        Forward pass through the CLIPAlignedShapeAsLatentModule, encoding shape, image, and text inputs into a common latent space.

        Args:
            surface (torch.FloatTensor): The shape input tensor.
            image (torch.FloatTensor): The image input tensor.
            text (torch.LongTensor): The text input tensor.

        Returns:
            embed_outputs (dict): A dictionary containing the embedding outputs for image, text, and shape, as well as the logit scale.
            shape_latents (torch.FloatTensor): The shape latents.
        """
        # Text embedding
        b = text.shape[0]  # Batch size
        text_tokens = rearrange(text, "b t l -> (b t) l")  # Flatten the text tensor for batch processing
        text_embed = self.encode_text_embed(text_tokens)  # Encode the text tokens
        text_embed = rearrange(text_embed, "(b t) d -> b t d", b=b)  # Reshape the text embedding back to its original shape
        text_embed = text_embed.mean(dim=1)  # Average the embeddings across templates
        text_embed = text_embed / text_embed.norm(dim=-1, keepdim=True)  # Normalize the text embedding

        # Image embedding 
        image_embed = self.encode_image_embed(image)  # Encode the image

        # Shape embedding
        shape_embed, shape_latents = self.encode_shape_embed(surface, return_latents=True)  # Encode the shape and return latents

        embed_outputs = {
            "image_embed": image_embed,  # Image embedding
            "text_embed": text_embed,  # Text embedding
            "shape_embed": shape_embed,  # Shape embedding
            "logit_scale": self.clip_model.logit_scale.exp()  # Logit scale for computing similarity
        }

        return embed_outputs, shape_latents  # Return the embedding outputs and shape latents
