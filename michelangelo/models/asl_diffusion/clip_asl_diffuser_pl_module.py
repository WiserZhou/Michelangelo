# -*- coding: utf-8 -*-

from omegaconf import DictConfig
from typing import List, Tuple, Dict, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only

from diffusers.schedulers import (
    DDPMScheduler,
    DDIMScheduler,
    KarrasVeScheduler,
    DPMSolverMultistepScheduler
)

from michelangelo.utils import instantiate_from_config
from michelangelo.models.tsal.tsal_base import AlignedShapeAsLatentPLModule
from michelangelo.models.asl_diffusion.inference_utils import ddim_sample

SchedulerType = Union[DDIMScheduler, KarrasVeScheduler, DPMSolverMultistepScheduler]

def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self

class ClipASLDiffuser(pl.LightningModule):
    """
    This class represents the ClipASLDiffuser model, which is a PyTorch Lightning module.
    It combines a first stage model for encoding surfaces, a conditional stage model for encoding conditions,
    and a diffusion model for generating samples.
    """
    first_stage_model: Optional[AlignedShapeAsLatentPLModule] = None
    """
    The first stage model is responsible for encoding surfaces into latent space.
    """
    cond_stage_model: Optional[Union[nn.Module, pl.LightningModule]] = None
    """
    The conditional stage model is responsible for encoding conditions such as text or images.
    """
    model: nn.Module
    """
    The diffusion model is the core model for generating samples based on the encoded surfaces and conditions.
    """

    def __init__(self, *,
                first_stage_config,
                cond_stage_config,
                denoiser_cfg,
                scheduler_cfg,
                optimizer_cfg,
                loss_cfg,
                first_stage_key: str = "surface",
                cond_stage_key: str = "image",
                scale_by_std: bool = False,
                z_scale_factor: float = 1.0,
                ckpt_path: Optional[str] = None,
                ignore_keys: Union[Tuple[str], List[str]] = ()):
        """
        Initializes the ClipASLDiffuser model with various configurations and parameters.

        Args:
            first_stage_config: Configuration for the first stage model.
            cond_stage_config: Configuration for the conditional stage model.
            denoiser_cfg: Configuration for the diffusion model.
            scheduler_cfg: Configuration for the scheduler.
            optimizer_cfg: Configuration for the optimizer.
            loss_cfg: Configuration for the loss function.
            first_stage_key: The key for accessing the first stage model's input in the batch.
            cond_stage_key: The key for accessing the conditional stage model's input in the batch.
            scale_by_std: Whether to scale the latent space by the standard deviation of the encodings.
            z_scale_factor: The factor to scale the latent space if scale_by_std is False.
            ckpt_path: The path to the checkpoint file for loading pre-trained weights.
            ignore_keys: Keys to ignore when loading pre-trained weights.
        """
        super().__init__()

        self.first_stage_key = first_stage_key
        self.cond_stage_key = cond_stage_key

        # 1. lazy initialize first stage
        self.instantiate_first_stage(first_stage_config)

        # 2. initialize conditional stage
        self.instantiate_cond_stage(cond_stage_config)

        # 3. diffusion model
        self.model = instantiate_from_config(
            denoiser_cfg, device=None, dtype=None
        )

        self.optimizer_cfg = optimizer_cfg

        # 4. scheduling strategy
        self.scheduler_cfg = scheduler_cfg

        self.noise_scheduler: DDPMScheduler = instantiate_from_config(scheduler_cfg.noise)
        self.denoise_scheduler: SchedulerType = instantiate_from_config(scheduler_cfg.denoise)

        # 5. loss configures
        self.loss_cfg = loss_cfg

        self.scale_by_std = scale_by_std
        if scale_by_std:
            self.register_buffer("z_scale_factor", torch.tensor(z_scale_factor))
        else:
            self.z_scale_factor = z_scale_factor

        self.ckpt_path = ckpt_path
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def instantiate_non_trainable_model(self, config):
        """
        Instantiates a non-trainable model from a given configuration.

        Args:
            config: The configuration for the model.

        Returns:
            A non-trainable model instance.
        """
        model = instantiate_from_config(config)
        model = model.eval()
        model.train = disabled_train
        for param in model.parameters():
            param.requires_grad = False

        return model

    def instantiate_first_stage(self, first_stage_config):
        """
        Instantiates the first stage model and sets it to only use the shape model.

        Args:
            first_stage_config: The configuration for the first stage model.
        """
        self.first_stage_model = self.instantiate_non_trainable_model(first_stage_config)
        # Michelangelo/michelangelo/models/tsal/clip_asl_module.py
        # Sets the CLIP model to None, effectively disabling it.
        self.first_stage_model.set_shape_model_only()

    def instantiate_cond_stage(self, cond_stage_config):
        """
        Instantiates the conditional stage model.

        Args:
            cond_stage_config: The configuration for the conditional stage model.
        """
        self.cond_stage_model = self.instantiate_non_trainable_model(cond_stage_config)

    def init_from_ckpt(self, path, ignore_keys=()):
        """
        Initializes the model from a checkpoint file, ignoring specified keys.

        Args:
            path: The path to the checkpoint file.
            ignore_keys: Keys to ignore when loading the checkpoint.
        """
        state_dict = torch.load(path, map_location="cpu")["state_dict"]

        keys = list(state_dict.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del state_dict[k]

        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
            print(f"Unexpected Keys: {unexpected}")

    @property
    def zero_rank(self):
        """
        Checks if the current process is the zero rank process.

        Returns:
            True if the current process is the zero rank process, False otherwise.
        """
        if self._trainer:
            zero_rank = self.trainer.local_rank == 0
        else:
            zero_rank = True

        return zero_rank

    def configure_optimizers(self) -> Tuple[List, List]:
        """
        Configures the optimizers and schedulers for the model.

        Returns:
            A tuple containing a list of optimizers and a list of schedulers.
        """
        lr = self.learning_rate

        trainable_parameters = list(self.model.parameters())
        if self.optimizer_cfg is None:
            optimizers = [torch.optim.AdamW(trainable_parameters, lr=lr, betas=(0.9, 0.99), weight_decay=1e-3)]
            schedulers = []
        else:
            optimizer = instantiate_from_config(self.optimizer_cfg.optimizer, params=trainable_parameters)
            scheduler_func = instantiate_from_config(
                self.optimizer_cfg.scheduler,
                max_decay_steps=self.trainer.max_steps,
                lr_max=lr
            )
            scheduler = {
                "scheduler": lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler_func.schedule),
                "interval": "step",
                "frequency": 1
            }
            optimizers = [optimizer]
            schedulers = [scheduler]

        return optimizers, schedulers

    @torch.no_grad()
    def encode_first_stage(self, surface: torch.FloatTensor, sample_posterior=True):
        """
        Encodes the surface into the latent space using the first stage model.

        Args:
            surface: The surface to encode.
            sample_posterior: Whether to sample from the posterior distribution.

        Returns:
            The encoded latent space.
        """
        z_q = self.first_stage_model.encode(surface, sample_posterior)
        z_q = self.z_scale_factor * z_q

        return z_q

    @torch.no_grad()
    def decode_first_stage(self, z_q: torch.FloatTensor, **kwargs):
        """
        Decodes the latent space back into the surface using the first stage model.

        Args:
            z_q: The latent space to decode.
            **kwargs: Additional keyword arguments for the decoding process.

        Returns:
            The decoded surface.
        """
        z_q = 1. / self.z_scale_factor * z_q
        latents = self.first_stage_model.decode(z_q, **kwargs)
        return latents

    @rank_zero_only
    @torch.no_grad()
    def on_train_batch_start(self, batch, batch_idx):
        """
        This method is called at the start of each training batch. It performs specific operations
        only for the very first batch of the first epoch, when no checkpoint is loaded. The purpose
        is to dynamically adjust the scaling factor for the latent space based on the standard deviation
        of the initial batch's encodings.

        Args:
            batch: The current batch being processed.
            batch_idx: The index of the current batch within the epoch.
        """
        # Check if it's the first batch of the first epoch and no checkpoint is loaded
        if self.scale_by_std and self.current_epoch == 0 and self.global_step == 0 \
                and batch_idx == 0 and self.ckpt_path is None:
            # Indicate that standard deviation rescaling is being used
            print("### USING STD-RESCALING ###")

            # Encode the first batch to get the latent space representation
            z_q = self.encode_first_stage(batch[self.first_stage_key])
            # Detach the tensor to prevent gradients from flowing back
            z = z_q.detach()

            # Remove any existing scaling factor buffer
            del self.z_scale_factor
            # Calculate the scaling factor as the reciprocal of the standard deviation of the flattened latent space
            self.register_buffer("z_scale_factor", 1. / z.flatten().std())
            # Log the new scaling factor value
            print(f"setting self.z_scale_factor to {self.z_scale_factor}")

            # Indicate that standard deviation rescaling has been applied
            print("### USING STD-RESCALING ###")

    def compute_loss(self, model_outputs, split):
        """
        This method calculates the loss based on the outputs from the model and the current batch split (train or validation).

        Args:
            model_outputs (dict): A dictionary containing the outputs from the model. The keys in this dictionary include:
                - x_0: The original input data.
                - noise: The noise added to the input data.
                - noise_prior: The prior noise distribution.
                - noise_pred: The predicted noise.
                - noise_pred_prior: The prior distribution of the predicted noise.
            split (str): A string indicating the current batch split, either 'train' or 'val'.

        Returns:
            A tuple containing the total loss and a dictionary of loss terms.
        """
        # Extract the predicted noise from the model outputs
        pred = model_outputs["pred"]

        # Determine the target for the loss calculation based on the prediction type
        if self.noise_scheduler.prediction_type == "epsilon":
            # If the prediction type is 'epsilon', the target is the actual noise
            target = model_outputs["noise"]
        elif self.noise_scheduler.prediction_type == "sample":
            # If the prediction type is 'sample', the target is the original input data
            target = model_outputs["x_0"]
        else:
            # If the prediction type is neither 'epsilon' nor 'sample', raise an exception
            raise NotImplementedError(f"Prediction Type: {self.noise_scheduler.prediction_type} not yet supported.")

        # Calculate the loss based on the specified loss type
        if self.loss_cfg.loss_type == "l1":
            # If the loss type is 'l1', use the mean absolute error
            simple = F.l1_loss(pred, target, reduction="mean")
        elif self.loss_cfg.loss_type in ["mse", "l2"]:
            # If the loss type is 'mse' or 'l2', use the mean squared error
            simple = F.mse_loss(pred, target, reduction="mean")
        else:
            # If the loss type is neither 'l1', 'mse', nor 'l2', raise an exception
            raise NotImplementedError(f"Loss Type: {self.loss_cfg.loss_type} not yet supported.")

        # The total loss is the simple loss calculated above
        total_loss = simple

        # Prepare a dictionary to store the loss terms
        loss_dict = {
            # Store the total loss under the key '{split}/total_loss'
            f"{split}/total_loss": total_loss.clone().detach(),
            # Store the simple loss under the key '{split}/simple'
            f"{split}/simple": simple.detach(),
        }

        # Return the total loss and the dictionary of loss terms
        return total_loss, loss_dict

    def forward(self, batch):
        # Encode the first stage input (e.g., surface) into latent space
        latents = self.encode_first_stage(batch[self.first_stage_key])
        # Encode the conditional stage input (e.g., image or text) into a conditional representation
        conditions = self.cond_stage_model.encode(batch[self.cond_stage_key])

        # Sample random noise to be added to the latents, with the same shape as the latents
        noise = torch.randn_like(latents)
        # Get the batch size from the shape of the latents
        bs = latents.shape[0]
        # Sample a random timestep for each batch element, within the range of training timesteps
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (bs,),
            device=latents.device,
        )
        # Ensure timesteps are of type long
        timesteps = timesteps.long()
        # Add noise to the latents based on the noise magnitude at each timestep
        noisy_z = self.noise_scheduler.add_noise(latents, noise, timesteps)

        # Perform a forward pass through the diffusion model, predicting the noise
        noise_pred = self.model(noisy_z, timesteps, conditions)

        # Prepare a dictionary to store the outputs from the diffusion model
        diffusion_outputs = {
            "x_0": noisy_z,  # The noisy latents
            "noise": noise,  # The original noise added to the latents
            "pred": noise_pred  # The predicted noise
        }

        # Return the outputs from the diffusion model
        return diffusion_outputs

    def training_step(self, batch: Dict[str, Union[torch.FloatTensor, List[str]]],
                      batch_idx: int, optimizer_idx: int = 0) -> torch.FloatTensor:
        """
        Performs a training step.

        Args:
            batch (dict): the batch sample, and it contains:
                - surface (torch.FloatTensor):
                - image (torch.FloatTensor): if provide, [bs, 3, h, w], item range [0, 1]
                - depth (torch.FloatTensor): if provide, [bs, 1, h, w], item range [-1, 1]
                - normal (torch.FloatTensor): if provide, [bs, 3, h, w], item range [-1, 1]
                - text (list of str):
            batch_idx: The index of the current batch.
            optimizer_idx: The index of the optimizer.

        Returns:
            The loss for the current batch.
        """
        diffusion_outputs = self(batch)

        loss, loss_dict = self.compute_loss(diffusion_outputs, "train")
        self.log_dict(loss_dict, prog_bar=True, logger=True, sync_dist=False, rank_zero_only=True)

        return loss

    def validation_step(self, batch: Dict[str, torch.FloatTensor],
                        batch_idx: int, optimizer_idx: int = 0) -> torch.FloatTensor:
        """
        Performs a validation step.

        Args:
            batch (dict): the batch sample, and it contains:
                - surface_pc (torch.FloatTensor): [n_pts, 4]
                - surface_feats (torch.FloatTensor): [n_pts, c]
                - text (list of str):
            batch_idx: The index of the current batch.
            optimizer_idx: The index of the optimizer.

        Returns:
            The loss for the current batch.
        """
        diffusion_outputs = self(batch)

        loss, loss_dict = self.compute_loss(diffusion_outputs, "val")
        self.log_dict(loss_dict, prog_bar=True, logger=True, sync_dist=False, rank_zero_only=True)

        return loss

    @torch.no_grad()
    def sample(self,
            batch: Dict[str, Union[torch.FloatTensor, List[str]]],
            sample_times: int = 1,
            steps: Optional[int] = None,
            guidance_scale: Optional[float] = None,
            eta: float = 0.0,
            return_intermediates: bool = False, **kwargs):
        """
        This method samples from the model. It generates samples based on the input batch and specified parameters.

        Args:
            batch (dict): The batch to sample from. It contains the input data for the model.
            sample_times (int, optional): The number of times to sample. Defaults to 1.
            steps (Optional[int], optional): The number of steps for the sampling process. Defaults to None.
            guidance_scale (Optional[float], optional): The guidance scale for the sampling process. Defaults to None.
            eta (float, optional): The eta value for the sampling process. Defaults to 0.0.
            return_intermediates (bool, optional): Whether to return intermediate samples. Defaults to False.
            **kwargs: Additional keyword arguments for the sampling process.

        Returns:
            A list of sampled outputs.
        """
        # If the number of steps is not provided, use the default number of inference steps from the scheduler configuration.
        if steps is None:
            steps = self.scheduler_cfg.num_inference_steps

        # If the guidance scale is not provided, use the default guidance scale from the scheduler configuration.
        if guidance_scale is None:
            guidance_scale = self.scheduler_cfg.guidance_scale
        # Determine if classifier-free guidance is needed based on the guidance scale.
        do_classifier_free_guidance = guidance_scale > 0

        # Extract the conditional input from the batch.
        xc = batch[self.cond_stage_key]

        # Encode the conditional input.
        cond = self.cond_stage_model(xc)

        # If classifier-free guidance is required, concatenate the unconditional embedding with the conditional encoding.
        if do_classifier_free_guidance:
            un_cond = self.cond_stage_model.unconditional_embedding(batch_size=len(xc))
            cond = torch.cat([un_cond, cond], dim=0)

        # Initialize an empty list to store the outputs.
        outputs = []
        # Initialize latents to None.
        latents = None

        # If intermediate samples are not required, sample multiple times and decode the final latent.
        if not return_intermediates:
            for _ in range(sample_times):
                # Perform the sampling process.
                sample_loop = ddim_sample(
                    self.denoise_scheduler,
                    self.model,
                    shape=self.first_stage_model.latent_shape,
                    cond=cond,
                    steps=steps,
                    guidance_scale=guidance_scale,
                    do_classifier_free_guidance=do_classifier_free_guidance,
                    device=self.device,
                    eta=eta,
                    disable_prog=not self.zero_rank
                )
                # Iterate over the samples and timestamps from the sampling loop.
                for sample, t in sample_loop:
                    # Update the latents with the current sample.
                    latents = sample
                # Decode the final latent and append it to the outputs.
                outputs.append(self.decode_first_stage(latents, **kwargs))
        else:
            # If intermediate samples are required, perform the sampling process once and decode at specified intervals.
            sample_loop = ddim_sample(
                self.denoise_scheduler,
                self.model,
                shape=self.first_stage_model.latent_shape,
                cond=cond,
                steps=steps,
                guidance_scale=guidance_scale,
                do_classifier_free_guidance=do_classifier_free_guidance,
                device=self.device,
                eta=eta,
                disable_prog=not self.zero_rank
            )

            # Calculate the iteration size to determine when to decode and append intermediate samples.
            iter_size = steps // sample_times
            i = 0
            for sample, t in sample_loop:
                # Update the latents with the current sample.
                latents = sample
                # If the current iteration is a multiple of the iteration size or the last iteration, decode and append.
                if i % iter_size == 0 or i == steps - 1:
                    outputs.append(self.decode_first_stage(latents, **kwargs))
                i += 1

        # Return the list of outputs.
        return outputs
