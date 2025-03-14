# -*- coding: utf-8 -*-

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from michelangelo.models.modules.checkpoint import checkpoint


def init_linear(l, stddev):
    """
    Initializes a linear layer with a normal distribution for weights and a constant for bias.

    Args:
        l (nn.Module): The linear layer to be initialized.
        stddev (float): The standard deviation for the normal distribution of weights.
    """
    # Initializes the weights of the linear layer with a normal distribution
    nn.init.normal_(l.weight, std=stddev)
    # If the linear layer has a bias term, initializes it with a constant value of 0.0
    if l.bias is not None:
        nn.init.constant_(l.bias, 0.0)


class MultiheadAttention(nn.Module):
    def __init__(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        n_ctx: int,
        width: int,
        heads: int,
        init_scale: float,
        qkv_bias: bool,
        flash: bool = False
    ):
        super().__init__()
        self.n_ctx = n_ctx
        self.width = width
        self.heads = heads
        self.c_qkv = nn.Linear(width, width * 3, bias=qkv_bias, device=device, dtype=dtype)
        self.c_proj = nn.Linear(width, width, device=device, dtype=dtype)
        self.attention = QKVMultiheadAttention(device=device, dtype=dtype, heads=heads, n_ctx=n_ctx, flash=flash)
        init_linear(self.c_qkv, init_scale)
        init_linear(self.c_proj, init_scale)

    def forward(self, x):
        x = self.c_qkv(x)
        x = checkpoint(self.attention, (x,), (), True)
        x = self.c_proj(x)
        return x


class QKVMultiheadAttention(nn.Module):
    def __init__(self, *, device: torch.device, dtype: torch.dtype, heads: int, n_ctx: int, flash: bool = False):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.heads = heads
        self.n_ctx = n_ctx
        self.flash = flash

    def forward(self, qkv):
        bs, n_ctx, width = qkv.shape
        attn_ch = width // self.heads // 3
        scale = 1 / math.sqrt(math.sqrt(attn_ch))
        qkv = qkv.view(bs, n_ctx, self.heads, -1)
        q, k, v = torch.split(qkv, attn_ch, dim=-1)

        if self.flash:
            out = F.scaled_dot_product_attention(q, k, v)
        else:
            weight = torch.einsum(
                "bthc,bshc->bhts", q * scale, k * scale
            )  # More stable with f16 than dividing afterwards
            wdtype = weight.dtype
            weight = torch.softmax(weight.float(), dim=-1).type(wdtype)
            out = torch.einsum("bhts,bshc->bthc", weight, v).reshape(bs, n_ctx, -1)

        return out


class ResidualAttentionBlock(nn.Module):
    def __init__(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        n_ctx: int,
        width: int,
        heads: int,
        init_scale: float = 1.0,
        qkv_bias: bool = True,
        flash: bool = False,
        use_checkpoint: bool = False
    ):
        super().__init__()

        self.use_checkpoint = use_checkpoint

        self.attn = MultiheadAttention(
            device=device,
            dtype=dtype,
            n_ctx=n_ctx,
            width=width,
            heads=heads,
            init_scale=init_scale,
            qkv_bias=qkv_bias,
            flash=flash
        )
        self.ln_1 = nn.LayerNorm(width, device=device, dtype=dtype)
        self.mlp = MLP(device=device, dtype=dtype, width=width, init_scale=init_scale)
        self.ln_2 = nn.LayerNorm(width, device=device, dtype=dtype)

    def _forward(self, x: torch.Tensor):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

    def forward(self, x: torch.Tensor):
        return checkpoint(self._forward, (x,), self.parameters(), self.use_checkpoint)


class MultiheadCrossAttention(nn.Module):
    """
    This module implements a multi-head cross-attention mechanism for transformer-based models.
    It processes input tensors through linear projections, multi-head attention, and a final linear projection.
    """
    def __init__(
        self,
        *,
        device: torch.device,  # The device to run the module on.
        dtype: torch.dtype,  # The data type to use for the module's parameters.
        width: int,  # The width of the transformer layers.
        heads: int,  # The number of attention heads in the transformer layers.
        init_scale: float,  # The scale factor for initializing the weights.
        qkv_bias: bool = True,  # Whether to use bias terms in the query, key, and value projections.
        flash: bool = False,  # Whether to use the flash attention mechanism.
        n_data: Optional[int] = None,  # The number of data points to process.
        data_width: Optional[int] = None,  # The width of the data to process. Defaults to the transformer width if not specified.
    ):
        super().__init__()
        self.n_data = n_data  # Store the number of data points.
        self.width = width  # Store the width of the transformer layers.
        self.heads = heads  # Store the number of attention heads.
        # If data_width is not specified, use the transformer width as the data width.
        self.data_width = width if data_width is None else data_width
        
        # Initialize linear layers for query, key-value, and projection.
        self.c_q = nn.Linear(width, width, bias=qkv_bias, device=device, dtype=dtype)
        self.c_kv = nn.Linear(self.data_width, width * 2, bias=qkv_bias, device=device, dtype=dtype)
        self.c_proj = nn.Linear(width, width, device=device, dtype=dtype)
        
        # Initialize the multi-head cross-attention module.
        self.attention = QKVMultiheadCrossAttention(
            device=device, dtype=dtype, heads=heads, n_data=n_data, flash=flash
        )
        
        # Initialize the linear layers with the specified scale factor.
        init_linear(self.c_q, init_scale)
        init_linear(self.c_kv, init_scale)
        init_linear(self.c_proj, init_scale)

    def forward(self, x: torch.Tensor, data: torch.Tensor):
        """
        Forward pass through the MultiheadCrossAttention module.

        Args:
            x (torch.Tensor): The input tensor to process.
            data (torch.Tensor): The data tensor to process.

        Returns:
            torch.Tensor: The processed output tensor.
        """
        # Project the input tensor for the query.
        x = self.c_q(x)
        # Project the data tensor for the key and value.
        data = self.c_kv(data)
        # Apply checkpointing to the attention module for memory efficiency.
        x = checkpoint(self.attention, (x, data), (), True)
        # Project the output of the attention module.
        x = self.c_proj(x)
        return x


class QKVMultiheadCrossAttention(nn.Module):
    """
    This module implements a multi-head cross-attention mechanism for transformer-based models.
    It supports both standard and flash attention mechanisms for computing attention weights.
    """
    def __init__(self, *, device: torch.device, dtype: torch.dtype, heads: int,
                flash: bool = False, n_data: Optional[int] = None):
        """
        Initializes the QKVMultiheadCrossAttention module.

        Args:
            device (torch.device): The device to run the module on.
            dtype (torch.dtype): The data type to use for the module's parameters.
            heads (int): The number of attention heads in the transformer layers.
            flash (bool, optional): Whether to use the flash attention mechanism. Defaults to False.
            n_data (Optional[int], optional): The number of data points to process. Defaults to None.
        """
        super().__init__()
        self.device = device  # Store the device for the module.
        self.dtype = dtype  # Store the data type for the module's parameters.
        self.heads = heads  # Store the number of attention heads.
        self.n_data = n_data  # Store the number of data points.
        self.flash = flash  # Store the flag for using the flash attention mechanism.

    def forward(self, q, kv):
        """
        Forward pass through the QKVMultiheadCrossAttention module.

        Args:
            q (torch.Tensor): The input tensor for the query.
            kv (torch.Tensor): The input tensor for the key and value.

        Returns:
            torch.Tensor: The processed output tensor.
        """
        _, n_ctx, _ = q.shape  # Extract the shape of the query tensor.
        bs, n_data, width = kv.shape  # Extract the shape of the key-value tensor.
        attn_ch = width // self.heads // 2  # Calculate the attention channel size.
        scale = 1 / math.sqrt(math.sqrt(attn_ch))  # Calculate the scaling factor for attention weights.
        q = q.view(bs, n_ctx, self.heads, -1)  # Reshape the query tensor for multi-head attention.
        kv = kv.view(bs, n_data, self.heads, -1)  # Reshape the key-value tensor for multi-head attention.
        k, v = torch.split(kv, attn_ch, dim=-1)  # Split the key-value tensor into key and value tensors.

        if self.flash:  # If using the flash attention mechanism.
            out = F.scaled_dot_product_attention(q, k, v)  # Compute attention using the flash mechanism.
        else:  # If not using the flash attention mechanism.
            weight = torch.einsum(
                "bthc,bshc->bhts", q * scale, k * scale  # Compute attention weights using einsum.
            )  # More stable with f16 than dividing afterwards
            wdtype = weight.dtype  # Store the data type of the weights.
            weight = torch.softmax(weight.float(), dim=-1).type(wdtype)  # Apply softmax to the weights.
            out = torch.einsum("bhts,bshc->bthc", weight, v).reshape(bs, n_ctx, -1)  # Compute the output using the weights and value.

        return out


class ResidualCrossAttentionBlock(nn.Module):
    """
    This module implements a residual cross-attention block for transformer-based models.
    It combines layer normalization, multi-head cross-attention, and a multi-layer perceptron (MLP) to process input tensors.
    """
    def __init__(
        self,
        *,
        device: Optional[torch.device],  # The device to run the module on.
        dtype: Optional[torch.dtype],  # The data type to use for the module's parameters.
        n_data: Optional[int] = None,  # The number of data points to process.
        width: int,  # The width of the transformer layers.
        heads: int,  # The number of attention heads in the transformer layers.
        data_width: Optional[int] = None,  # The width of the data to process. Defaults to the transformer width if not specified.
        init_scale: float = 0.25,  # The scale factor for initializing the weights.
        qkv_bias: bool = True,  # Whether to use bias terms in the query, key, and value projections.
        flash: bool = False  # Whether to use the flash attention mechanism.
    ):
        super().__init__()

        # If data_width is not specified, set it to the transformer width.
        if data_width is None:
            data_width = width

        # Initialize the multi-head cross-attention module.
        self.attn = MultiheadCrossAttention(
            device=device,
            dtype=dtype,
            n_data=n_data,
            width=width,
            heads=heads,
            data_width=data_width,
            init_scale=init_scale,
            qkv_bias=qkv_bias,
            flash=flash,
        )
        # Initialize layer normalization modules for input tensors.
        self.ln_1 = nn.LayerNorm(width, device=device, dtype=dtype)
        self.ln_2 = nn.LayerNorm(data_width, device=device, dtype=dtype)
        # Initialize the MLP module for further processing.
        self.mlp = MLP(device=device, dtype=dtype, width=width, init_scale=init_scale)
        # Initialize another layer normalization module for the output of the MLP.
        self.ln_3 = nn.LayerNorm(width, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, data: torch.Tensor):
        """
        Forward pass through the ResidualCrossAttentionBlock module.

        Args:
            x (torch.Tensor): The input tensor to process.
            data (torch.Tensor): The data tensor to use for cross-attention.

        Returns:
            torch.Tensor: The processed output tensor.
        """
        # Apply layer normalization to the input tensors.
        x = x + self.attn(self.ln_1(x), self.ln_2(data))
        # Apply layer normalization to the output of the MLP.
        x = x + self.mlp(self.ln_3(x))
        return x


class MLP(nn.Module):
    """
    A simple multi-layer perceptron (MLP) module for further processing in the transformer block.
    This module consists of two linear layers with a GELU activation function in between.
    """
    def __init__(self, *,
                 device: Optional[torch.device],
                 dtype: Optional[torch.dtype],
                 width: int,
                 init_scale: float):
        """
        Initializes the MLP module.

        Args:
            device (Optional[torch.device]): The device to use for the module.
            dtype (Optional[torch.dtype]): The data type to use for the module.
            width (int): The width of the input and output tensors.
            init_scale (float): The scale factor for initializing the weights of the linear layers.
        """
        super().__init__()
        self.width = width
        # The first linear layer expands the input width by a factor of 4.
        self.c_fc = nn.Linear(width, width * 4, device=device, dtype=dtype)
        # The second linear layer projects the output back to the original width.
        self.c_proj = nn.Linear(width * 4, width, device=device, dtype=dtype)
        # The GELU activation function is used for non-linearity.
        self.gelu = nn.GELU()
        # Initialize the weights of the linear layers with the specified scale factor.
        init_linear(self.c_fc, init_scale)
        init_linear(self.c_proj, init_scale)

    def forward(self, x):
        """
        Forward pass through the MLP module.

        Args:
            x (torch.Tensor): The input tensor to the MLP module.

        Returns:
            torch.Tensor: The output tensor from the MLP module.
        """
        # The forward pass applies the first linear layer, followed by GELU activation, and then the second linear layer.
        return self.c_proj(self.gelu(self.c_fc(x)))


class Transformer(nn.Module):
    """
    This class represents a Transformer model, which is a type of neural network architecture particularly well-suited for sequential data.
    It is composed of multiple ResidualAttentionBlock modules, each of which applies self-attention and a feed-forward network to the input.
    """
    def __init__(
        self,
        *,
        device: Optional[torch.device],  # The device to use for the Transformer model.
        dtype: Optional[torch.dtype],  # The data type to use for the Transformer model.
        n_ctx: int,  # The context size, which is the maximum sequence length the model can process.
        width: int,  # The width of the input and output tensors.
        layers: int,  # The number of ResidualAttentionBlock layers in the Transformer.
        heads: int,  # The number of attention heads in each ResidualAttentionBlock.
        init_scale: float = 0.25,  # The scale factor for initializing the weights of the linear layers.
        qkv_bias: bool = True,  # A flag indicating whether to use bias terms in the query, key, and value linear layers.
        flash: bool = False,  # A flag indicating whether to use the flash attention mechanism.
        use_checkpoint: bool = False  # A flag indicating whether to use gradient checkpointing for memory efficiency.
    ):
        super().__init__()
        self.n_ctx = n_ctx  # Store the context size.
        self.width = width  # Store the width of the input and output tensors.
        self.layers = layers  # Store the number of layers.
        # Initialize a ModuleList to hold the ResidualAttentionBlock modules.
        self.resblocks = nn.ModuleList(
            [
                ResidualAttentionBlock(
                    device=device,
                    dtype=dtype,
                    n_ctx=n_ctx,
                    width=width,
                    heads=heads,
                    init_scale=init_scale,
                    qkv_bias=qkv_bias,
                    flash=flash,
                    use_checkpoint=use_checkpoint
                )
                for _ in range(layers)  # Create a ResidualAttentionBlock for each layer.
            ]
        )

    def forward(self, x: torch.Tensor):
        # Iterate through each ResidualAttentionBlock and apply it to the input.
        for block in self.resblocks:
            x = block(x)
        return x
