# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from typing import Optional
from einops import repeat
import math

from michelangelo.models.modules import checkpoint
from michelangelo.models.modules.embedder import FourierEmbedder
from michelangelo.models.modules.distributions import DiagonalGaussianDistribution
from michelangelo.models.modules.transformer_blocks import (
    ResidualCrossAttentionBlock,
    Transformer
)

from .tsal_base import ShapeAsLatentModule

class CrossAttentionEncoder(nn.Module):
    """
    This module implements a cross-attention encoder for processing point cloud data.
    It combines Fourier embedding with cross-attention and self-attention mechanisms to encode point cloud features.
    """

    def __init__(self, *,
                device: Optional[torch.device],
                dtype: Optional[torch.dtype],
                num_latents: int,
                fourier_embedder: FourierEmbedder,
                point_feats: int,
                width: int,
                heads: int,
                layers: int,
                init_scale: float = 0.25,
                qkv_bias: bool = True,
                flash: bool = False,
                use_ln_post: bool = False,
                use_checkpoint: bool = False):
        """
        Initializes the CrossAttentionEncoder module.

        Args:
            device (Optional[torch.device]): The device to run the module on.
            dtype (Optional[torch.dtype]): The data type to use for the module's parameters.
            num_latents (int): The number of latent variables to use for encoding.
            fourier_embedder (FourierEmbedder): The Fourier embedder module to use for embedding point cloud coordinates.
            point_feats (int): The number of features per point in the point cloud.
            width (int): The width of the transformer layers.
            heads (int): The number of attention heads in the transformer layers.
            layers (int): The number of transformer layers.
            init_scale (float, optional): The scale factor for initializing the weights. Defaults to 0.25.
            qkv_bias (bool, optional): Whether to use bias terms in the query, key, and value projections. Defaults to True.
            flash (bool, optional): Whether to use the flash attention mechanism. Defaults to False.
            use_ln_post (bool, optional): Whether to use layer normalization after the self-attention block. Defaults to False.
            use_checkpoint (bool, optional): Whether to use checkpointing for the forward pass. Defaults to False.
        """
        super().__init__()

        self.use_checkpoint = use_checkpoint
        self.num_latents = num_latents

        # Initialize the query parameters for the cross-attention block
        self.query = nn.Parameter(torch.randn((num_latents, width), device=device, dtype=dtype) * 0.02)

        # Store the Fourier embedder module
        self.fourier_embedder = fourier_embedder

        # Define a linear layer to project the input data to the transformer width
        self.input_proj = nn.Linear(self.fourier_embedder.out_dim + point_feats, width, device=device, dtype=dtype)

        # Define the cross-attention block
        self.cross_attn = ResidualCrossAttentionBlock(
            device=device,
            dtype=dtype,
            width=width,
            heads=heads,
            init_scale=init_scale,
            qkv_bias=qkv_bias,
            flash=flash,
        )

        # Define the self-attention transformer
        self.self_attn = Transformer(
            device=device,
            dtype=dtype,
            n_ctx=num_latents,
            width=width,
            layers=layers,
            heads=heads,
            init_scale=init_scale,
            qkv_bias=qkv_bias,
            flash=flash,
            use_checkpoint=False
        )

        # Optionally define a layer normalization module for post-processing
        if use_ln_post:
            self.ln_post = nn.LayerNorm(width, dtype=dtype, device=device)
        else:
            self.ln_post = None

    def _forward(self, pc, feats):
        """
        Forward pass through the CrossAttentionEncoder module.

        Args:
            pc (torch.FloatTensor): The point cloud data [B, N, 3].
            feats (torch.FloatTensor or None): Optional features for each point [B, N, C].

        Returns:
            latents (torch.FloatTensor): The encoded latent variables.
            pc (torch.FloatTensor): The original point cloud data.
        """
        bs = pc.shape[0]

        # Apply Fourier embedding to the point cloud coordinates
        data = self.fourier_embedder(pc)
        # If additional features are provided, concatenate them to the embedded coordinates
        if feats is not None:
            data = torch.cat([data, feats], dim=-1)
        # Project the data to the transformer width
        data = self.input_proj(data)

        # Repeat the query parameters for each batch element
        query = repeat(self.query, "m c -> b m c", b=bs)
        # Perform cross-attention encoding
        latents = self.cross_attn(query, data)
        # Perform self-attention encoding
        latents = self.self_attn(latents)

        # Optionally apply layer normalization
        if self.ln_post is not None:
            latents = self.ln_post(latents)

        return latents, pc

    def forward(self, pc: torch.FloatTensor, feats: Optional[torch.FloatTensor] = None):
        """
        Forward pass through the CrossAttentionEncoder module with optional checkpointing.

        Args:
            pc (torch.FloatTensor): The point cloud data [B, N, 3].
            feats (torch.FloatTensor or None): Optional features for each point [B, N, C].

        Returns:
            dict: A dictionary containing the encoded latent variables and the original point cloud data.
        """
        return checkpoint(self._forward, (pc, feats), self.parameters(), self.use_checkpoint)

class CrossAttentionDecoder(nn.Module):
    """
    This module implements a cross-attention decoder for processing queries and latents.
    It applies Fourier embedding to queries, projects them to the transformer width,
    performs cross-attention decoding, applies layer normalization, and finally projects
    the output to the desired number of channels.
    """

    def __init__(self, *,
                device: Optional[torch.device],
                dtype: Optional[torch.dtype],
                num_latents: int,
                out_channels: int,
                fourier_embedder: FourierEmbedder,
                width: int,
                heads: int,
                init_scale: float = 0.25,
                qkv_bias: bool = True,
                flash: bool = False,
                use_checkpoint: bool = False):
        """
        Initializes the CrossAttentionDecoder module.

        Args:
            device (Optional[torch.device]): The device to run the module on.
            dtype (Optional[torch.dtype]): The data type to use for the module's parameters.
            num_latents (int): The number of latent variables to process.
            out_channels (int): The number of output channels.
            fourier_embedder (FourierEmbedder): The Fourier embedder module to use.
            width (int): The width of the transformer layers.
            heads (int): The number of attention heads in the transformer layers.
            init_scale (float, optional): The scale factor for initializing the weights. Defaults to 0.25.
            qkv_bias (bool, optional): Whether to use bias terms in the query, key, and value projections. Defaults to True.
            flash (bool, optional): Whether to use the flash attention mechanism. Defaults to False.
            use_checkpoint (bool, optional): Whether to use gradient checkpointing for memory efficiency. Defaults to False.
        """
        super().__init__()

        self.use_checkpoint = use_checkpoint  # Store the flag for using gradient checkpointing.
        self.fourier_embedder = fourier_embedder  # Store the Fourier embedder module.

        # Initialize a linear layer to project queries to the transformer width.
        self.query_proj = nn.Linear(self.fourier_embedder.out_dim, width, device=device, dtype=dtype)

        # Initialize a residual cross-attention block for decoding.
        self.cross_attn_decoder = ResidualCrossAttentionBlock(
            device=device,
            dtype=dtype,
            n_data=num_latents,
            width=width,
            heads=heads,
            init_scale=init_scale,
            qkv_bias=qkv_bias,
            flash=flash
        )

        # Initialize a layer normalization module for post-processing.
        self.ln_post = nn.LayerNorm(width, device=device, dtype=dtype)
        # Initialize a linear layer to project the output to the desired number of channels.
        self.output_proj = nn.Linear(width, out_channels, device=device, dtype=dtype)

    def _forward(self, queries: torch.FloatTensor, latents: torch.FloatTensor):
        """
        Forward pass through the CrossAttentionDecoder module without checkpointing.

        Args:
            queries (torch.FloatTensor): The input queries tensor.
            latents (torch.FloatTensor): The input latents tensor.

        Returns:
            torch.FloatTensor: The processed output tensor.
        """
        # Apply Fourier embedding to the queries.
        queries = self.query_proj(self.fourier_embedder(queries))
        # Perform cross-attention decoding.
        x = self.cross_attn_decoder(queries, latents)
        # Apply layer normalization.
        x = self.ln_post(x)
        # Project the output to the desired number of channels.
        x = self.output_proj(x)
        return x

    def forward(self, queries: torch.FloatTensor, latents: torch.FloatTensor):
        """
        Forward pass through the CrossAttentionDecoder module with optional checkpointing.

        Args:
            queries (torch.FloatTensor): The input queries tensor.
            latents (torch.FloatTensor): The input latents tensor.

        Returns:
            torch.FloatTensor: The processed output tensor.
        """
        # Apply checkpointing to the forward pass if enabled.
        return checkpoint(self._forward, (queries, latents), self.parameters(), self.use_checkpoint)


class ShapeAsLatentPerceiver(ShapeAsLatentModule):
    def __init__(self, *,
                device: Optional[torch.device],
                dtype: Optional[torch.dtype],
                num_latents: int,
                point_feats: int = 0,
                embed_dim: int = 0,
                num_freqs: int = 8,
                include_pi: bool = True,
                width: int,
                heads: int,
                num_encoder_layers: int,
                num_decoder_layers: int,
                init_scale: float = 0.25,
                qkv_bias: bool = True,
                flash: bool = False,
                use_ln_post: bool = False,
                use_checkpoint: bool = False):

        # Initialize the parent class
        super().__init__()

        # Store the use_checkpoint flag
        self.use_checkpoint = use_checkpoint

        # Store the number of latents
        self.num_latents = num_latents
        # Initialize a FourierEmbedder
        self.fourier_embedder = FourierEmbedder(num_freqs=num_freqs, include_pi=include_pi)

        # Initialize the scale factor for initializing the weights
        init_scale = init_scale * math.sqrt(1.0 / width)
        
        # Initialize the CrossAttentionEncoder
        self.encoder = CrossAttentionEncoder(
            device=device,
            dtype=dtype,
            fourier_embedder=self.fourier_embedder,
            num_latents=num_latents,
            point_feats=point_feats,
            width=width,
            heads=heads,
            layers=num_encoder_layers,
            init_scale=init_scale,
            qkv_bias=qkv_bias,
            flash=flash,
            use_ln_post=use_ln_post,
            use_checkpoint=use_checkpoint
        )

        # Store the embed_dim
        self.embed_dim = embed_dim
        # If embed_dim is greater than 0, initialize the VAE embed
        if embed_dim > 0:
            self.pre_kl = nn.Linear(width, embed_dim * 2, device=device, dtype=dtype)
            self.post_kl = nn.Linear(embed_dim, width, device=device, dtype=dtype)
            self.latent_shape = (num_latents, embed_dim)
        else:
            self.latent_shape = (num_latents, width)

        # Initialize the Transformer
        self.transformer = Transformer(
            device=device,
            dtype=dtype,
            n_ctx=num_latents,
            width=width,
            layers=num_decoder_layers,
            heads=heads,
            init_scale=init_scale,
            qkv_bias=qkv_bias,
            flash=flash,
            use_checkpoint=use_checkpoint
        )

        # Initialize the geometry decoder
        self.geo_decoder = CrossAttentionDecoder(
            device=device,
            dtype=dtype,
            fourier_embedder=self.fourier_embedder,
            out_channels=1,
            num_latents=num_latents,
            width=width,
            heads=heads,
            init_scale=init_scale,
            qkv_bias=qkv_bias,
            flash=flash,
            use_checkpoint=use_checkpoint
        )

    def encode(self,
            pc: torch.FloatTensor,
            feats: Optional[torch.FloatTensor] = None,
            sample_posterior: bool = True):
        """
        Encode the input point cloud and features.

        Args:
            pc (torch.FloatTensor): The input point cloud tensor.
            feats (torch.FloatTensor or None): The input features tensor.
            sample_posterior (bool): A flag indicating whether to sample the posterior distribution.

        Returns:
            latents (torch.FloatTensor): The encoded latents tensor.
            center_pos (torch.FloatTensor or None): The center position tensor.
            posterior (DiagonalGaussianDistribution or None): The posterior distribution.
        """

        # Encode the input point cloud and features and get the latents and point cloud
        latents, center_pos = self.encoder(pc, feats)

        # If embed_dim is greater than 0, initialize the VAE embed
        # Initialize the posterior distribution
        posterior = None
        if self.embed_dim > 0:
            # Calculate the moments of the latent variables
            moments = self.pre_kl(latents)
            # Create a diagonal Gaussian distribution with the moments
            posterior = DiagonalGaussianDistribution(moments, feat_dim=-1)

            # If sample_posterior is True, sample from the posterior distribution
            if sample_posterior:
                latents = posterior.sample()
            else:
                # If sample_posterior is False, use the mode of the posterior distribution
                latents = posterior.mode()

        return latents, center_pos, posterior

    def decode(self, latents: torch.FloatTensor):
        # Decode the input latents
        latents = self.post_kl(latents)
        return self.transformer(latents)

    def query_geometry(self, queries: torch.FloatTensor, latents: torch.FloatTensor):
        # Query the geometry
        logits = self.geo_decoder(queries, latents).squeeze(-1)
        return logits

    def forward(self,
                pc: torch.FloatTensor,
                feats: torch.FloatTensor,
                volume_queries: torch.FloatTensor,
                sample_posterior: bool = True):
        """
        Forward pass through the ShapeAsLatentPerceiver module.

        Args:
            pc (torch.FloatTensor): The input point cloud tensor.
            feats (torch.FloatTensor or None): The input features tensor.
            volume_queries (torch.FloatTensor): The input volume queries tensor.
            sample_posterior (bool): A flag indicating whether to sample the posterior distribution.

        Returns:
            logits (torch.FloatTensor): The queried logits tensor.
            center_pos (torch.FloatTensor): The center position tensor.
            posterior (DiagonalGaussianDistribution or None): The posterior distribution.
        """

        # Encode the input point cloud and features
        latents, center_pos, posterior = self.encode(pc, feats, sample_posterior=sample_posterior)

        # Decode the input latents
        latents = self.decode(latents)
        # Query the geometry
        logits = self.query_geometry(volume_queries, latents)

        return logits, center_pos, posterior
    
class AlignedShapeLatentPerceiver(ShapeAsLatentPerceiver):
    """
    This class extends the ShapeAsLatentPerceiver to provide additional functionality for aligned shape latent perception.
    It is designed to handle the encoding of shape inputs into a latent space, allowing for various applications such as shape analysis and generation.
    """

    def __init__(self, *,
                device: Optional[torch.device],
                dtype: Optional[torch.dtype],
                num_latents: int,
                point_feats: int = 0,
                embed_dim: int = 0,
                num_freqs: int = 8,
                include_pi: bool = True,
                width: int,
                heads: int,
                num_encoder_layers: int,
                num_decoder_layers: int,
                init_scale: float = 0.25,
                qkv_bias: bool = True,
                flash: bool = False,
                use_ln_post: bool = False,
                use_checkpoint: bool = False):

        """
        Initializes the AlignedShapeLatentPerceiver with various parameters for configuration.

        Args:
            device (Optional[torch.device]): The device to use for computations.
            dtype (Optional[torch.dtype]): The data type to use for tensors.
            num_latents (int): The number of latents to use for encoding.
            point_feats (int, optional): The number of point features to use. Defaults to 0.
            embed_dim (int, optional): The dimensionality of the embedding space. Defaults to 0.
            num_freqs (int, optional): The number of frequencies to use for Fourier embedding. Defaults to 8.
            include_pi (bool, optional): A flag indicating whether to include pi in the Fourier embedding. Defaults to True.
            width (int): The width of the model.
            heads (int): The number of attention heads.
            num_encoder_layers (int): The number of layers in the encoder.
            num_decoder_layers (int): The number of layers in the decoder.
            init_scale (float, optional): The scale factor for initializing weights. Defaults to 0.25.
            qkv_bias (bool, optional): A flag indicating whether to use bias in the query, key, and value projections. Defaults to True.
            flash (bool, optional): A flag indicating whether to use the flash attention mechanism. Defaults to False.
            use_ln_post (bool, optional): A flag indicating whether to use layer normalization in the posterior. Defaults to False.
            use_checkpoint (bool, optional): A flag indicating whether to use checkpointing. Defaults to False.
        """

        super().__init__(
            device=device,
            dtype=dtype,
            num_latents=1 + num_latents,
            point_feats=point_feats,
            embed_dim=embed_dim,
            num_freqs=num_freqs,
            include_pi=include_pi,
            width=width,
            heads=heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            init_scale=init_scale,
            qkv_bias=qkv_bias,
            flash=flash,
            use_ln_post=use_ln_post,
            use_checkpoint=use_checkpoint
        )

        self.width = width

    def encode(self,
               pc: torch.FloatTensor,
               feats: Optional[torch.FloatTensor] = None,
               sample_posterior: bool = True):
        """
        Encodes the input point cloud and features into a latent space.

        Args:
            pc (torch.FloatTensor): The input point cloud tensor of shape [B, N, 3], where B is the batch size, N is the number of points, and 3 is the dimensionality of the points.
            feats (torch.FloatTensor or None, optional): The input features tensor of shape [B, N, c], where c is the number of features. Defaults to None.
            sample_posterior (bool, optional): A flag indicating whether to sample from the posterior distribution. Defaults to True.

        Returns:
            shape_embed (torch.FloatTensor): The shape embedding tensor.
            kl_embed (torch.FloatTensor): The KL embedding tensor.
            posterior (DiagonalGaussianDistribution or None): The posterior distribution or None if embed_dim is 0.
        """

        shape_embed, latents = self.encode_latents(pc, feats)
        kl_embed, posterior = self.encode_kl_embed(latents, sample_posterior)

        return shape_embed, kl_embed, posterior

    def encode_latents(self,
                       pc: torch.FloatTensor,
                       feats: Optional[torch.FloatTensor] = None):

        """
        Encodes the input point cloud and features into latents.

        Args:
            pc (torch.FloatTensor): The input point cloud tensor of shape [B, N, 3], where B is the batch size, N is the number of points, and 3 is the dimensionality of the points.
            feats (torch.FloatTensor or None, optional): The input features tensor of shape [B, N, c], where c is the number of features. Defaults to None.

        Returns:
            shape_embed (torch.FloatTensor): The shape embedding tensor.
            latents (torch.FloatTensor): The latents tensor.
        """

        x, _ = self.encoder(pc, feats)

        shape_embed = x[:, 0]
        latents = x[:, 1:]

        return shape_embed, latents

    def encode_kl_embed(self, latents: torch.FloatTensor, sample_posterior: bool = True):
        """
        Encodes the latents into a KL embedding and optionally samples from the posterior distribution.

        Args:
            latents (torch.FloatTensor): The latents tensor.
            sample_posterior (bool, optional): A flag indicating whether to sample from the posterior distribution. Defaults to True.

        Returns:
            kl_embed (torch.FloatTensor): The KL embedding tensor.
            posterior (DiagonalGaussianDistribution or None): The posterior distribution or None if embed_dim is 0.
        """

        posterior = None
        if self.embed_dim > 0:
            moments = self.pre_kl(latents)
            posterior = DiagonalGaussianDistribution(moments, feat_dim=-1)

            if sample_posterior:
                kl_embed = posterior.sample()
            else:
                kl_embed = posterior.mode()
        else:
            kl_embed = latents

        return kl_embed, posterior

    def forward(self,
                pc: torch.FloatTensor,
                feats: torch.FloatTensor,
                volume_queries: torch.FloatTensor,
                sample_posterior: bool = True):
        """
        Forward pass through the AlignedShapeLatentPerceiver.

        Args:
            pc (torch.FloatTensor): The input point cloud tensor of shape [B, N, 3], where B is the batch size, N is the number of points, and 3 is the dimensionality of the points.
            feats (torch.FloatTensor): The input features tensor of shape [B, N, c], where c is the number of features.
            volume_queries (torch.FloatTensor): The input volume queries tensor of shape [B, P, 3], where P is the number of queries.
            sample_posterior (bool, optional): A flag indicating whether to sample from the posterior distribution. Defaults to True.

        Returns:
            shape_embed (torch.FloatTensor): The shape embedding tensor of shape [B, projection_dim].
            logits (torch.FloatTensor): The logits tensor of shape [B, M], where M is the number of output dimensions.
            shape_embed (torch.FloatTensor): [B, projection_dim]
            logits (torch.FloatTensor): [B, M]
            posterior (DiagonalGaussianDistribution or None).

        """

        shape_embed, kl_embed, posterior = self.encode(pc, feats, sample_posterior=sample_posterior)

        latents = self.decode(kl_embed)
        logits = self.query_geometry(volume_queries, latents)

        return shape_embed, logits, posterior
