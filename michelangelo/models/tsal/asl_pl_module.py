# -*- coding: utf-8 -*-

from typing import List, Tuple, Dict, Optional
from omegaconf import DictConfig

import torch
import torch.nn.functional as F
from torch.optim import lr_scheduler
import pytorch_lightning as pl
from typing import Union
from functools import partial
import numpy

from michelangelo.utils import instantiate_from_config

from .inference_utils import extract_geometry
from .tsal_base import (
    AlignedShapeAsLatentModule,
    ShapeAsLatentModule,
    Latent2MeshOutput,
    AlignedMeshOutput
)

class AlignedShapeAsLatentPLModule(pl.LightningModule):
    """
    This class extends the pl.LightningModule to handle the training, validation, and inference of the AlignedShapeAsLatentModule.
    It is designed to be used in the first stage of the ASL Diffuser model.
    """

    def __init__(self, *,
                shape_module_cfg,
                aligned_module_cfg,
                loss_cfg,
                optimizer_cfg: Optional[DictConfig] = None,
                ckpt_path: Optional[str] = None,
                ignore_keys: Union[Tuple[str], List[str]] = ()):
        """
        Initializes the AlignedShapeAsLatentPLModule with the shape module configuration, aligned module configuration, 
        loss configuration, optimizer configuration, checkpoint path, and ignore keys.

        Args:
            shape_module_cfg: The configuration for the shape module.
            aligned_module_cfg: The configuration for the aligned module.
            loss_cfg: The configuration for the loss function.
            optimizer_cfg (DictConfig, optional): The configuration for the optimizer. Defaults to None.
            ckpt_path (str, optional): The path to the checkpoint. Defaults to None.
            ignore_keys (Union[Tuple[str], List[str]], optional): The keys to ignore when loading the checkpoint. Defaults to ().
        """
        super().__init__()

        shape_model: ShapeAsLatentModule = instantiate_from_config(
            shape_module_cfg, device=None, dtype=None
        )
        self.model: AlignedShapeAsLatentModule = instantiate_from_config(
            aligned_module_cfg, shape_model=shape_model
        )

        self.loss = instantiate_from_config(loss_cfg)

        self.optimizer_cfg = optimizer_cfg

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

        self.save_hyperparameters()

    def set_shape_model_only(self):
        """
        Sets the shape model to None, effectively disabling it.
        """
        self.model.set_shape_model_only()

    @property
    def latent_shape(self):
        """
        Returns the latent shape of the shape model.
        """
        return self.model.shape_model.latent_shape

    @property
    def zero_rank(self):
        """
        Returns a flag indicating whether the current rank is zero.
        """
        if self._trainer:
            zero_rank = self.trainer.local_rank == 0
        else:
            zero_rank = True

        return zero_rank

    def init_from_ckpt(self, path, ignore_keys=()):
        """
        Initializes the model from a checkpoint.

        Args:
            path (str): The path to the checkpoint.
            ignore_keys (Union[Tuple[str], List[str]], optional): The keys to ignore when loading the checkpoint. Defaults to ().
        """
        # Load the state dictionary from the checkpoint file
        # 它的作用是确保在加载模型时，只有指定的全局对象可以被访问。
        with torch.serialization.safe_globals([numpy.core.multiarray.scalar]):
            state_dict = torch.load(path, map_location="cpu", weights_only=False)["state_dict"]

        # Get the keys of the state dictionary
        keys = list(state_dict.keys())
        # Iterate over the keys
        for k in keys:
            # Iterate over the ignore keys
            for ik in ignore_keys:
                # If the key starts with an ignore key, delete the key from the state dictionary
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del state_dict[k]

        # Load the state dictionary into the model, allowing for missing and unexpected keys
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        # Print the number of missing and unexpected keys
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        # If there are missing keys, print them
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
        # If there are unexpected keys, print them
        if len(unexpected) > 0:
            print(f"Unexpected Keys: {unexpected}")

    def configure_optimizers(self) -> Tuple[List, List]:
        """
        Configures the optimizers and schedulers for training.

        Returns:
            Tuple[List, List]: A tuple containing a list of optimizers and a list of schedulers.
        """
        # Retrieve the learning rate from the model's configuration
        lr = self.learning_rate

        # Collect all trainable parameters from the model
        trainable_parameters = list(self.model.parameters())

        # If no optimizer configuration is provided, use the default AdamW optimizer
        if self.optimizer_cfg is None:
            optimizers = [torch.optim.AdamW(trainable_parameters, lr=lr, betas=(0.9, 0.99), weight_decay=1e-3)]
            schedulers = []
        else:
            # Instantiate the optimizer from the provided configuration
            optimizer = instantiate_from_config(self.optimizer_cfg.optimizer, params=trainable_parameters)
            # Instantiate the scheduler function from the provided configuration
            scheduler_func = instantiate_from_config(
                self.optimizer_cfg.scheduler,
                max_decay_steps=self.trainer.max_steps,
                lr_max=lr
            )
            # Create a scheduler configuration with the instantiated scheduler function
            scheduler = {
                "scheduler": lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler_func.schedule),
                "interval": "step",
                "frequency": 1
            }
            # Prepare the optimizer and scheduler for training
            optimizers = [optimizer]
            schedulers = [scheduler]

        # Return the configured optimizers and schedulers for training
        return optimizers, schedulers

    def forward(self,
                surface: torch.FloatTensor,
                image: torch.FloatTensor,
                text: torch.FloatTensor,
                volume_queries: torch.FloatTensor):
        """
        Forwards the input through the model.

        Args:
            surface (torch.FloatTensor): The surface input tensor.
            image (torch.FloatTensor): The image input tensor.
            text (torch.FloatTensor): The text input tensor.
            volume_queries (torch.FloatTensor): The volume queries input tensor.

        Returns:
            Tuple[Dict, torch.FloatTensor, Dict]: A tuple containing a dictionary of embed outputs, the logits, and a dictionary of posteriors.
        """
        # Forward pass through the model with the input tensors
        embed_outputs, shape_z = self.model(surface, image, text)

        # Encode the shape embedding and calculate the posterior
        shape_zq, posterior = self.model.shape_model.encode_kl_embed(shape_z)

        # Decode the latent shape
        latents = self.model.shape_model.decode(shape_zq)

        # Query the geometry with the volume queries and the decoded latents
        logits = self.model.shape_model.query_geometry(volume_queries, latents)

        # Return the embed outputs, logits, and posterior
        return embed_outputs, logits, posterior

    def encode(self, surface: torch.FloatTensor, sample_posterior=True):
        """
        Encodes the surface input into a latent space using the shape model.

        Args:
            surface (torch.FloatTensor): The surface input tensor.
            sample_posterior (bool, optional): Whether to sample the posterior. Defaults to True.

        Returns:
            torch.FloatTensor: The encoded latent shape.
        """
        pc = surface[..., 0:3]
        feats = surface[..., 3:6]

        shape_embed, shape_zq, posterior = self.model.shape_model.encode(
            pc=pc, feats=feats, sample_posterior=sample_posterior
        )

        return shape_zq

    def decode(self,
            z_q,
            bounds: Union[Tuple[float], List[float], float] = 1.1,
            octree_depth: int = 7,
            num_chunks: int = 10000) -> List[Latent2MeshOutput]:
        """
        Decodes the latent shape into a mesh.

        Args:
            z_q: The latent shape.
            bounds: The bounds of the mesh. Defaults to 1.1.
            octree_depth: The depth of the octree. Defaults to 7.
            num_chunks: The number of chunks to divide the dense grid into for processing. Defaults to 10000.

        Returns:
            List[Latent2MeshOutput]: The list of mesh outputs.
        """

        latents = self.model.shape_model.decode(z_q)  # latents: [bs, num_latents, dim]
        outputs = self.latent2mesh(latents, bounds=bounds, octree_depth=octree_depth, num_chunks=num_chunks)

        return outputs

    def training_step(self, batch: Dict[str, torch.FloatTensor],
                    batch_idx: int, optimizer_idx: int = 0) -> torch.FloatTensor:
        """
        Handles a training step.

        Args:
            batch (dict): the batch sample, and it contains:
                - surface (torch.FloatTensor): [bs, n_surface, (3 + input_dim)]
                - image (torch.FloatTensor): [bs, 3, 224, 224]
                - text (torch.FloatTensor): [bs, num_templates, 77]
                - geo_points (torch.FloatTensor): [bs, n_pts, (3 + 1)]
            batch_idx (int): The index of the batch.
            optimizer_idx (int, optional): The index of the optimizer. Defaults to 0.

        Returns:
            torch.FloatTensor: The loss.
        """

        surface = batch["surface"]
        image = batch["image"]
        text = batch["text"]

        volume_queries = batch["geo_points"][..., 0:3]
        shape_labels = batch["geo_points"][..., -1]

        embed_outputs, shape_logits, posteriors = self(surface, image, text, volume_queries)

        aeloss, log_dict_ae = self.loss(
            **embed_outputs,
            posteriors=posteriors,
            shape_logits=shape_logits,
            shape_labels=shape_labels,
            split="train"
        )

        self.log_dict(log_dict_ae, prog_bar=True, logger=True, batch_size=shape_logits.shape[0],
                    sync_dist=False, rank_zero_only=True)

        return aeloss

    def validation_step(self, batch: Dict[str, torch.FloatTensor], batch_idx: int) -> torch.FloatTensor:
        """
        Handles a validation step.

        Args:
            batch (dict): The batch sample.
            batch_idx (int): The index of the batch.

        Returns:
            torch.FloatTensor: The loss.
        """

        surface = batch["surface"]
        image = batch["image"]
        text = batch["text"]

        volume_queries = batch["geo_points"][..., 0:3]
        shape_labels = batch["geo_points"][..., -1]

        embed_outputs, shape_logits, posteriors = self(surface, image, text, volume_queries)

        aeloss, log_dict_ae = self.loss(
            **embed_outputs,
            posteriors=posteriors,
            shape_logits=shape_logits,
            shape_labels=shape_labels,
            split="val"
        )
        self.log_dict(log_dict_ae, prog_bar=True, logger=True, batch_size=shape_logits.shape[0],
                    sync_dist=False, rank_zero_only=True)

        return aeloss

    def visual_alignment(self,
                        surface: torch.FloatTensor,
                        image: torch.FloatTensor,
                        text: torch.FloatTensor,
                        description: Optional[List[str]] = None,
                        bounds: Union[Tuple[float], List[float]] = (-1.25, -1.25, -1.25, 1.25, 1.25, 1.25),
                        octree_depth: int = 7,
                        num_chunks: int = 10000) -> List[AlignedMeshOutput]:
        """
        Visualizes the alignment of the model.

        Args:
            surface: The surface input tensor.
            image: The image input tensor.
            text: The text input tensor.
            description: The description of the alignment. Defaults to None.
            bounds: The bounds of the mesh. Defaults to (-1.25, -1.25, -1.25, 1.25, 1.25, 1.25).
            octree_depth: The depth of the octree. Defaults to 7.
            num_chunks: The number of chunks to divide the dense grid into for processing. Defaults to 10000.

        Returns:
            List[AlignedMeshOutput]: The list of aligned mesh outputs.
        """

        outputs = []

        device = surface.device
        bs = surface.shape[0]

        embed_outputs, shape_z = self.model(surface, image, text)

        # calculate the similarity
        image_embed = embed_outputs["image_embed"]
        text_embed = embed_outputs["text_embed"]
        shape_embed = embed_outputs["shape_embed"]

        # normalized features
        shape_embed = F.normalize(shape_embed, dim=-1, p=2)
        text_embed = F.normalize(text_embed, dim=-1, p=2)
        image_embed = F.normalize(image_embed, dim=-1, p=2)

        # B x B
        shape_text_similarity = (100.0 * shape_embed @ text_embed.T).softmax(dim=-1)

        # B x B
        shape_image_similarity = (100.0 * shape_embed @ image_embed.T).softmax(dim=-1)

        # shape reconstruction
        shape_zq, posterior = self.model.shape_model.encode_kl_embed(shape_z)
        latents = self.model.shape_model.decode(shape_zq)
        geometric_func = partial(self.model.shape_model.query_geometry, latents=latents)

        # 2. decode geometry
        mesh_v_f, has_surface = extract_geometry(
            geometric_func=geometric_func,
            device=device,
            batch_size=bs,
            bounds=bounds,
            octree_depth=octree_depth,
            num_chunks=num_chunks,
            disable=not self.zero_rank
        )

        # 3. decode texture
        for i, ((mesh_v, mesh_f), is_surface) in enumerate(zip(mesh_v_f, has_surface)):
            if not is_surface:
                outputs.append(None)
                continue

            out = AlignedMeshOutput()
            out.mesh_v = mesh_v
            out.mesh_f = mesh_f
            out.surface = surface[i].cpu().numpy()
            out.image = image[i].cpu().numpy()
            if description is not None:
                out.text = description[i]
            out.shape_text_similarity = shape_text_similarity[i, i]
            out.shape_image_similarity = shape_image_similarity[i, i]

            outputs.append(out)

        return outputs

    def latent2mesh(self,
                    latents: torch.FloatTensor,
                    bounds: Union[Tuple[float], List[float], float] = 1.1,
                    octree_depth: int = 7,
                    num_chunks: int = 10000) -> List[Latent2MeshOutput]:
        """
        Converts the latent shape into a mesh.

        Args:
            latents: [bs, num_latents, dim]
            bounds: The bounds of the mesh. Defaults to 1.1.
            octree_depth: The depth of the octree. Defaults to 7.
            num_chunks: The number of chunks to divide the dense grid into for processing. Defaults to 10000.

        Returns:
            List[Latent2MeshOutput]: The list of mesh outputs.
        """

        outputs = []

        geometric_func = partial(self.model.shape_model.query_geometry, latents=latents)

        # 2. decode geometry
        device = latents.device
        mesh_v_f, has_surface = extract_geometry(
            geometric_func=geometric_func,
            device=device,
            batch_size=len(latents),
            bounds=bounds,
            octree_depth=octree_depth,
            num_chunks=num_chunks,
            disable=not self.zero_rank
        )

        # 3. decode texture
        for i, ((mesh_v, mesh_f), is_surface) in enumerate(zip(mesh_v_f, has_surface)):
            if not is_surface:
                outputs.append(None)
                continue

            out = Latent2MeshOutput()
            out.mesh_v = mesh_v
            out.mesh_f = mesh_f

            outputs.append(out)

        return outputs
