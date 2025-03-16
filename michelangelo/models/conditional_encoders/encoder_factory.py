# -*- coding: utf-8 -*-
import os

import torch
import torch.nn as nn
from torchvision import transforms
from transformers import CLIPModel, CLIPTokenizer
from collections import OrderedDict
import clip

from michelangelo.data.transforms import RandomResize


class AbstractEncoder(nn.Module):
    embedding_dim: int

    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError

# This is a class embedder module that is used to embed class information.
class ClassEmbedder(nn.Module):
    # Initialize the ClassEmbedder module with the specified embedding dimension, number of classes, and key.
    def __init__(self, embed_dim, n_classes=1000, key="class"):
        super().__init__()
        self.key = key
        # Create an embedding layer with the specified number of classes and embedding dimension.
        self.embedding = nn.Embedding(n_classes, embed_dim)

    # Define the forward pass of the ClassEmbedder module.
    def forward(self, batch, key=None):
        # If the key is not provided, use the default key.
        if key is None:
            key = self.key
        # Extract the class information from the batch.
        c = batch[key][:, None]
        # Embed the class information using the embedding layer.
        c = self.embedding(c)
        # Return the embedded class information.
        return c


class FrozenCLIPTextEmbedder(AbstractEncoder):
    """Uses the CLIP transformer encoder for text (from Hugging Face)"""

    def __init__(
        self,
        version="openai/clip-vit-large-patch14",
        tokenizer_version=None,
        device="cuda",
        max_length=77,
        zero_embedding_radio: float = 0.1,
    ):
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(tokenizer_version or version)

        self.device = device
        self.max_length = max_length
        self.zero_embedding_radio = zero_embedding_radio

        self.clip_dict = OrderedDict()
        self.clip_name = os.path.split(version)[-1]

        transformer = CLIPModel.from_pretrained(version).text_model

        for param in transformer.parameters():
            param.requires_grad = False
        self.clip_dict[self.clip_name] = transformer

        self._move_flag = False

    @property
    def clip(self):
        return self.clip_dict[self.clip_name]

    def move(self):
        if self._move_flag:
            return

        self.clip_dict[self.clip_name] = self.clip_dict[self.clip_name].to(self.device)
        self._move_flag = True

    def unconditional_embedding(self, batch_size):
        empty_text = [""] * batch_size
        empty_z = self.forward(empty_text)
        return empty_z

    def forward(self, text):
        self.move()

        batch_encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_length=True,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt",
        )

        tokens = batch_encoding["input_ids"].to(self.device)
        outputs = self.clip(input_ids=tokens)

        z = outputs.last_hidden_state
        return z

    def encode(self, text):
        batch_size = len(text)
        batch_mask = torch.rand((batch_size,))
        for i in range(batch_size):
            if batch_mask[i] < self.zero_embedding_radio:
                text[i] = ""

        return self(text)

class FrozenAlignedCLIPTextEmbedder(AbstractEncoder):
    """
    This class utilizes the CLIP transformer encoder for text processing, specifically designed for text embedding tasks.
    It leverages the Hugging Face library for model loading and tokenization.
    """

    def __init__(
        self,
        version="openai/clip-vit-large-patch14",
        tokenizer_version=None,
        device="cuda",
        max_length=77,
        zero_embedding_radio: float = 0.1,
    ):
        """
        Initializes the FrozenAlignedCLIPTextEmbedder with specified parameters.

        Args:
            version (str, optional): The version of the CLIP model to use. Defaults to "openai/clip-vit-large-patch14".
            tokenizer_version (str, optional): The version of the tokenizer to use. Defaults to None, which uses the same version as the model.
            device (str, optional): The device to move the model to. Defaults to "cuda".
            max_length (int, optional): The maximum length of the input text. Defaults to 77.
            zero_embedding_radio (float, optional): The probability of setting an embedding to zero during training. Defaults to 0.1.
        """
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(tokenizer_version or version)

        self.device = device
        self.max_length = max_length
        self.zero_embedding_radio = zero_embedding_radio

        self.clip_dict = OrderedDict()
        self.clip_name = os.path.split(version)[-1]

        transformer = CLIPModel.from_pretrained(version).text_model

        # Freeze the model parameters to prevent training
        for param in transformer.parameters():
            param.requires_grad = False
        self.clip_dict[self.clip_name] = transformer

        self._move_flag = False

    @property
    def clip(self):
        """
        Returns the CLIP model instance.
        """
        return self.clip_dict[self.clip_name]

    def move(self):
        """
        Moves the model to the specified device and sets the move flag.
        """
        if self._move_flag:
            return

        self.clip_dict[self.clip_name] = self.clip_dict[self.clip_name].to(self.device)
        self._move_flag = True

    def unconditional_embedding(self, batch_size):
        """
        Generates an unconditional embedding tensor of zeros with the specified batch size.

        Args:
            batch_size (int): The size of the batch.

        Returns:
            torch.Tensor: The unconditional embedding tensor.
        """
        empty_text = [""] * batch_size
        empty_z = self.forward(empty_text)
        return empty_z

    def forward(self, text):
        """
        Forward pass through the model for text embedding.

        Args:
            text (list[str]): A list of text strings to be embedded.

        Returns:
            torch.Tensor: The embedded text tensor.
        """
        self.move()

        batch_encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_length=True,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt",
        )

        tokens = batch_encoding["input_ids"].to(self.device)
        outputs = self.clip(input_ids=tokens)

        z = outputs.last_hidden_state
        return z

    def encode(self, text):
        """
        Encodes the input text with a chance of setting some embeddings to zero.

        Args:
            text (list[str]): A list of text strings to be encoded.

        Returns:
            torch.Tensor: The encoded text tensor.
        """
        batch_size = len(text)
        batch_mask = torch.rand((batch_size,))
        for i in range(batch_size):
            if batch_mask[i] < self.zero_embedding_radio:
                text[i] = ""

        return self(text)


class FrozenCLIPImageEmbedder(AbstractEncoder):
    """Uses the CLIP transformer encoder for text (from Hugging Face)"""

    def __init__(
            self,
            version="./pretrained/clip-vit-large-patch14",
            device="cuda",
            zero_embedding_radio=0.1,
            normalize_embedding=True,
            num_projection_vector=0,
            linear_mapping_bias=True,
            reverse_visual_projection=False,
    ):
        super().__init__()

        self.device = device

        self.clip_dict = OrderedDict()
        self.clip_name = os.path.split(version)[-1]

        clip_model = CLIPModel.from_pretrained(version)
        clip_model.text_model = None
        clip_model.text_projection = None
        clip_model = clip_model.eval()
        for param in self.parameters():
            param.requires_grad = False
        self.clip_dict[self.clip_name] = clip_model

        self.transform = transforms.Compose(
            [
                transforms.Resize(224, transforms.InterpolationMode.BICUBIC, antialias=True),
                transforms.CenterCrop(224),  # crop a (224, 224) square
                transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                ),
            ]
        )
        self.zero_embedding_radio = zero_embedding_radio

        self.num_projection_vector = num_projection_vector
        self.reverse_visual_projection = reverse_visual_projection
        self.normalize_embedding = normalize_embedding

        embedding_dim = (
            clip_model.visual_projection.in_features
            if reverse_visual_projection
            else clip_model.visual_projection.out_features
        )
        self.embedding_dim = embedding_dim
        if self.num_projection_vector > 0:
            self.projection = nn.Linear(
                embedding_dim,
                clip_model.visual_projection.out_features * num_projection_vector,
                bias=linear_mapping_bias,
            )
            nn.init.normal_(self.projection.weight, std=embedding_dim ** -0.5)

        self._move_flag = False

    @property
    def clip(self):
        return self.clip_dict[self.clip_name]

    def unconditional_embedding(self, batch_size):
        zero = torch.zeros(
            batch_size,
            1,
            self.embedding_dim,
            device=self.device,
            dtype=self.clip.visual_projection.weight.dtype,
        )
        if self.num_projection_vector > 0:
            zero = self.projection(zero).view(batch_size, self.num_projection_vector, -1)
        return zero

    def forward(self, image, value_range=(-1, 1), zero_embedding_radio=0):
        if value_range is not None:
            low, high = value_range
            image = (image - low) / (high - low)

        image = image.to(self.device, dtype=self.clip.visual_projection.weight.dtype)

        if self.reverse_visual_projection:
            z = self.clip.vision_model(self.transform(image))[1]
        else:
            z = self.clip.get_image_features(self.transform(image))

        if self.normalize_embedding:
            z = z / z.norm(dim=-1, keepdim=True)
        if z.ndim == 2:
            z = z.unsqueeze(dim=-2)

        if zero_embedding_radio > 0:
            mask = torch.rand((len(image), 1, 1), device=z.device, dtype=z.dtype) < zero_embedding_radio
            z = z * mask.to(z)

        if self.num_projection_vector > 0:
            z = self.projection(z).view(len(image), self.num_projection_vector, -1)

        return z

    def move(self):
        if self._move_flag:
            return

        self.clip_dict[self.clip_name] = self.clip_dict[self.clip_name].to(self.device)
        self._move_flag = True

    def encode(self, image):
        self.move()
        return self(image, zero_embedding_radio=self.zero_embedding_radio)


class FrozenCLIPImageGridEmbedder(AbstractEncoder):
    """
    This class defines a FrozenCLIPImageGridEmbedder model, which is a PyTorch module that encapsulates the functionality of encoding images 
    using a frozen CLIP model with a specific version and projection vectors. It supports various precision modes, normalization, 
    and clipping options.
    """
    def __init__(
            self,
            version="./pretrained/clip-vit-large-patch14", # openai/clip-vit-large-patch14
            device="cuda",
            zero_embedding_radio=0.1,
    ):
        """
        Initializes the FrozenCLIPImageGridEmbedder model with the specified parameters.

        Args:
            version (str, optional): The version of the CLIP model to use. Defaults to "openai/clip-vit-large-patch14".
            device (str, optional): The device to use for computations. Defaults to "cuda".
            zero_embedding_radio (float, optional): The probability of zeroing out embeddings during training. Defaults to 0.1.
        """
        super().__init__()

        self.device = device

        # Initialize an ordered dictionary to store the CLIP model
        self.clip_dict = OrderedDict()
        # Extract the model name from the version string
        self.clip_name = os.path.split(version)[-1]

        # Load the CLIP model with the specified version and set it to evaluation mode
        clip_model: CLIPModel = CLIPModel.from_pretrained(version)
        clip_model.text_model = None
        clip_model.text_projection = None
        clip_model = clip_model.eval()
        # Freeze all model parameters
        for param in self.parameters():
            param.requires_grad = False
        # Store the model in the dictionary
        self.clip_dict[self.clip_name] = clip_model

        # Define the image transformation pipeline
        self.transform = transforms.Compose(
            [
                # Resize the image to 224x224 with bilinear interpolation and antialiasing
                transforms.Resize(224, transforms.InterpolationMode.BILINEAR, antialias=True),
                # Center crop the image to a 224x224 square
                transforms.CenterCrop(224), 
                # Normalize the image with mean and standard deviation
                transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                ),
            ]
        )
        # Store the zero embedding radio
        self.zero_embedding_radio = zero_embedding_radio
        # Get the embedding dimension from the CLIP model
        self.embedding_dim = clip_model.vision_embed_dim

        # Flag to track if the model has been moved to the device
        self._move_flag = False

    @property
    def clip(self):
        """
        Returns the CLIP model instance.
        """
        return self.clip_dict[self.clip_name]

    def move(self):
        """
        Moves the CLIP model to the specified device if it has not been moved already.
        """
        if self._move_flag:
            return

        self.clip_dict[self.clip_name] = self.clip_dict[self.clip_name].to(self.device)
        self._move_flag = True

    def unconditional_embedding(self, batch_size):
        """
        Generates an unconditional embedding tensor of zeros with the specified batch size.

        Args:
            batch_size (int): The batch size of the embedding tensor.

        Returns:
            torch.Tensor: The unconditional embedding tensor.
        """
        zero = torch.zeros(
            batch_size,
            self.clip.vision_model.embeddings.num_positions,
            self.embedding_dim,
            device=self.device,
            dtype=self.clip.visual_projection.weight.dtype,
        )
        return zero

    def forward(self, image, value_range=(-1, 1), zero_embedding_radio=0):
        """
        Forward pass through the FrozenCLIPImageGridEmbedder model.

        Args:
            image (torch.Tensor): The input image tensor.
            value_range (tuple, optional): The value range to normalize the image. Defaults to (-1, 1).
            zero_embedding_radio (float, optional): The probability of zeroing out embeddings during training. Defaults to 0.

        Returns:
            torch.Tensor: The encoded image embedding tensor.
        """
        self.move()

        if value_range is not None:
            low, high = value_range
            # Normalize the input image based on the provided value range
            image = (image - low) / (high - low)

        # Move the image to the device and convert to the appropriate dtype
        image = image.to(self.device, dtype=self.clip.visual_projection.weight.dtype)

        # Encode the image using the CLIP model
        z = self.clip.vision_model(self.transform(image)).last_hidden_state

        if zero_embedding_radio > 0:
            # Generate a mask for randomly zeroing out embeddings
            mask = torch.rand((len(image), 1, 1), device=z.device, dtype=z.dtype) >= zero_embedding_radio
            # Apply the mask to the embeddings
            z = z * mask.to(z)

        return z

    def encode(self, image):
        """
        Encodes the input image using the FrozenCLIPImageGridEmbedder model.

        Args:
            image (torch.Tensor): The input image tensor.

        Returns:
            torch.Tensor: The encoded image embedding tensor.
        """
        return self(image, zero_embedding_radio=self.zero_embedding_radio)

class MoECLIPImageEncoder(nn.Module):
    """
    This class defines a MoECLIPImageEncoder model, which is a PyTorch module that encapsulates the functionality of encoding images 
    using multiple CLIP models with different versions and projection vectors. It supports various precision modes, normalization, 
    and clipping options.
    """
    def __init__(
            self,
            versions,
            hidden_state_dim,
            num_projection_vector=8,
            zero_embedding_radio=0.1,
            device="cuda",
            precision="fp16",
            normalize=False,
            clip_max=0,
            transform_type="base",
            argument_p=0.2,
    ):
        """
        Initializes the MoECLIPImageEncoder model with the specified parameters.

        Args:
            versions (str or tuple): The version(s) of the CLIP model to use. If a single string is provided, it is converted to a tuple.
            hidden_state_dim (int): The dimension of the hidden state in the projection layer.
            num_projection_vector (int, optional): The number of projection vectors to use. Defaults to 8.
            zero_embedding_radio (float, optional): The probability of zeroing out embeddings during training. Defaults to 0.1.
            device (str, optional): The device to use for computations. Defaults to "cuda".
            precision (str, optional): The precision mode to use. Can be "fp16", "fp32", or "bf16". Defaults to "fp16".
            normalize (bool, optional): Whether to normalize the embeddings. Defaults to False.
            clip_max (float, optional): The maximum value to clip embeddings to. Defaults to 0.
            transform_type (str, optional): The type of transformation to apply to images. Can be "base" or "crop_blur_resize". Defaults to "base".
            argument_p (float, optional): The probability of applying random transformations. Defaults to 0.2.
        """
        super().__init__()

        self.device = torch.device(device)
        self.hidden_state_dim = hidden_state_dim
        self.zero_embedding_radio = zero_embedding_radio
        self.num_projection_vector = num_projection_vector
        self.dtype = dict(fp16=torch.float16, fp32=torch.float32, bf16=torch.bfloat16)[precision]
        self.normalize = normalize
        self.clip_max = clip_max

        # Define the image transformation pipeline based on the specified transform_type
        # This section dynamically configures the image transformation pipeline based on the 'transform_type' parameter.
        # It supports two types of transformations: 'base' and 'crop_blur_resize'.
        if transform_type == "base":
            # For 'base' transformation, the pipeline consists of resizing, center cropping, and normalization.
            self.transform = transforms.Compose(
                [
                    transforms.Resize(224, transforms.InterpolationMode.BICUBIC, antialias=True),  # Resize the image to 224x224 with bicubic 
                    # interpolation and antialiasing.
                    transforms.CenterCrop(224),  # Crop a 224x224 square from the center of the image.
                    transforms.Normalize(
                        mean=[0.48145466, 0.4578275, 0.40821073],  # Normalize the image with specified mean values.
                        std=[0.26862954, 0.26130258, 0.27577711],  # Normalize the image with specified standard deviation values.
                    ),
                ]
            )
        elif transform_type == "crop_blur_resize":
            # For 'crop_blur_resize' transformation, the pipeline includes resizing, center cropping, random resized cropping, Gaussian blur, 
            # random resizing, and normalization.
            self.transform = transforms.Compose(
                [
                    transforms.Resize(224, transforms.InterpolationMode.BICUBIC, antialias=True),  # Resize the image to 224x224 with bicubic 
                    # interpolation and antialiasing.
                    transforms.CenterCrop(224),  # Crop a 224x224 square from the center of the image.
                    transforms.RandomApply(
                        transforms=[
                            transforms.RandomResizedCrop(
                                size=224,  # Crop a random size between 0.8 and 1.0 of the original size.
                                scale=(0.8, 1.0),  # Scale factor range.
                                ratio=(0.99, 1.01),  # Aspect ratio range.
                                interpolation=transforms.InterpolationMode.BICUBIC,  # Interpolation mode.
                            ),
                        ],
                        p=argument_p,  # Probability of applying this transformation.
                    ),
                    transforms.RandomApply(
                        transforms=[
                            transforms.GaussianBlur(kernel_size=9, sigma=(0.1, 5)),  # Apply Gaussian blur with a kernel size of 9 and sigma 
                            # between 0.1 and 5.
                        ],
                        p=argument_p,  # Probability of applying this transformation.
                    ),
                    transforms.RandomApply(
                        transforms=[
                            RandomResize(size=224, resize_radio=(0.2, 1)),  # Resize the image to a random size between 0.2 and 1 of the original size.
                        ],
                        p=argument_p,  # Probability of applying this transformation.
                    ),
                    transforms.Normalize(
                        mean=[0.48145466, 0.4578275, 0.40821073],  # Normalize the image with specified mean values.
                        std=[0.26862954, 0.26130258, 0.27577711],  # Normalize the image with specified standard deviation values.
                    ),
                ]
            )
        else:
            # If the specified 'transform_type' is neither 'base' nor 'crop_blur_resize', raise a ValueError.
            raise ValueError(f"invalid {transform_type=}")

        # Convert the versions parameter to a tuple if it's a single string
        if isinstance(versions, str):
            versions = (versions,)

        # 如果直接把clips定位为当前类的子module，1. 会在保存ckp时存无用的多个权重。 2. pl会调用to，导致layer_norm的权重也被转换成fp16
        clips = OrderedDict()

        for v in versions:
            # 因为clips不是子module，直接指定device="cuda"会错误地导致clip模型权重都被放到cuda:0上。
            # Load the CLIP model with the specified version, device, and settings.
            clips[v], _ = clip.load(name=v, device="cpu", jit=False, download_root=None)
            # Remove the transformer attribute from the loaded CLIP model to avoid unnecessary computations.
            delattr(clips[v], "transformer")
            # Set the CLIP model to evaluation mode to disable gradient computation.
            clips[v].eval()
            # Disable gradient computation for the CLIP model to prevent backpropagation.
            clips[v].requires_grad_(False)

        # Calculate the total hidden dimension across all CLIP models
        self.clips_hidden_dim = sum(clips[v].ln_final.weight.size(0) for v in clips)

        # Define the projection layer based on the number of projection vectors
        if self.num_projection_vector == 0:
            self.projection = nn.Identity()
        else:
            self.projection = nn.Linear(self.clips_hidden_dim, hidden_state_dim * self.num_projection_vector, bias=True)
            self.projection.to(dtype=self.dtype)
            nn.init.normal_(self.projection.weight, std=self.clips_hidden_dim ** -0.5)

        self.clips = clips

        self._move_flag = False

    def move(self):
        """
        Moves the model to the specified device and converts the weights to the specified precision.
        """
        if self._move_flag:
            return

        def convert_weights(model: nn.Module):
            """
            Converts applicable model parameters to the specified precision.
            """
            def _convert_weights_to_fp16(l):
                if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
                    l.weight.data = l.weight.data.type(self.dtype)
                    if l.bias is not None:
                        l.bias.data = l.bias.data.type(self.dtype)

                if isinstance(l, nn.MultiheadAttention):
                    for attr in [
                        *[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]],
                        "in_proj_bias",
                        "bias_k",
                        "bias_v",
                    ]:
                        tensor = getattr(l, attr)
                        if tensor is not None:
                            tensor.data = tensor.data.type(self.dtype)

                for name in ["text_projection", "proj"]:
                    if hasattr(l, name):
                        attr = getattr(l, name)
                        if attr is not None:
                            attr.data = attr.data.type(self.dtype)

            model.apply(_convert_weights_to_fp16)

        for k in self.clips:
            self.clips[k].to(self.device)
            convert_weights(self.clips[k])  # fp32 -> self.dtype
        self._move_flag = True

    def unconditional_embedding(self, batch_size=None):
        """
        Generates an unconditional embedding tensor of zeros with the specified batch size and shape.
        """
        zero = torch.zeros(
            batch_size,
            self.clips_hidden_dim,
            device=self.device,
            dtype=self.dtype,
        )
        if self.num_projection_vector > 0:
            zero = self.projection(zero).view(batch_size, self.num_projection_vector, -1)
        return zero

    def convert_embedding(self, z):
        """
        Converts the input embedding tensor to the desired shape and precision.
        """
        if self.num_projection_vector > 0:
            z = self.projection(z.type(self.projection.weight.dtype)).view(len(z), self.num_projection_vector, -1)
        return z

    def forward(self, image, value_range=(-1, 1), zero_embedding_radio=0):
        """
        Forward pass through the MoECLIPImageEncoder model.

        Args:
            image (torch.Tensor): The input image tensor.
            value_range (tuple, optional): The value range to normalize the image. Defaults to (-1, 1).
            zero_embedding_radio (float, optional): The probability of zeroing out embeddings during training. Defaults to 0.

        Returns:
            torch.Tensor: The encoded image embedding tensor.
        """
        # Normalize the input image based on the provided value range
        if value_range is not None:
            low, high = value_range
            image = (image - low) / (high - low)

        # Apply the predefined transformation to the input image
        image = self.transform(image)

        # Encode the image using each clip model without gradients
        with torch.no_grad():
            embs = []
            for v in self.clips:
                x = self.clips[v].encode_image(image)
                # Normalize the encoded embedding if required
                if self.normalize:
                    x = x / x.norm(p=2, dim=-1, keepdim=True) * (x.size(-1) ** 0.5)
                    # Clip the normalized embedding if clip_max is set
                    if self.clip_max > 0:
                        x = x.clamp(-self.clip_max, self.clip_max)
                embs.append(x)

            # Concatenate all encoded embeddings
            z = torch.cat(embs, dim=-1)
            # Normalize the final concatenated embedding if required
            if self.normalize:
                z /= z.size(-1) ** 0.5

        # Randomly zero out embeddings during training if zero_embedding_radio is set
        if zero_embedding_radio > 0:
            mask = torch.rand((len(image), 1, 1), device=z.device, dtype=z.dtype) >= zero_embedding_radio
            z = z * mask.to(z)

        # Project the final embedding to the desired dimension if required
        if self.num_projection_vector > 0:
            z = self.projection(z).view(len(image), self.num_projection_vector, -1)
        return z

    def encode(self, image):
        """
        Encodes the input image using the MoECLIPImageEncoder model.

        Args:
            image (torch.Tensor): The input image tensor.

        Returns:
            torch.Tensor: The encoded image embedding tensor.
        """
        self.move()
        return self(image, zero_embedding_radio=self.zero_embedding_radio)
