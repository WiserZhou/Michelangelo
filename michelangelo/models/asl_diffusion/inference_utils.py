# -*- coding: utf-8 -*-

import torch
from tqdm import tqdm
from typing import Tuple, List, Union, Optional
from diffusers.schedulers import DDIMScheduler


__all__ = ["ddim_sample"]


def ddim_sample(ddim_scheduler: DDIMScheduler,
                diffusion_model: torch.nn.Module,
                shape: Union[List[int], Tuple[int]],
                cond: torch.FloatTensor,
                steps: int,
                eta: float = 0.0,
                guidance_scale: float = 3.0,
                do_classifier_free_guidance: bool = True,
                generator: Optional[torch.Generator] = None,
                device: torch.device = "cuda:0",
                disable_prog: bool = True):

    # Assert that the number of steps is positive
    assert steps > 0, f"{steps} must > 0."

    # Initialize latents based on the condition and shape
    bsz = cond.shape[0]
    if do_classifier_free_guidance:
        # Adjust batch size for classifier-free guidance
        bsz = bsz // 2

    # Generate random noise for the initial latents
    latents = torch.randn(
        (bsz, *shape),
        generator=generator,
        device=cond.device,
        dtype=cond.dtype,
    )
    # Scale the initial noise by the standard deviation required by the scheduler
    latents = latents * ddim_scheduler.init_noise_sigma
    # Set the timesteps for the scheduler
    ddim_scheduler.set_timesteps(steps)
    timesteps = ddim_scheduler.timesteps.to(device)
    # Prepare extra keyword arguments for the scheduler step
    # eta (η) is only used with the DDIMScheduler, and its value is between [0, 1]
    extra_step_kwargs = {
        "eta": eta,
        "generator": generator
    }

    # Reverse process for sampling
    for i, t in enumerate(tqdm(timesteps, disable=disable_prog, desc="DDIM Sampling:", leave=False)):
        # Expand the latents if we are doing classifier free guidance
        latent_model_input = (
            torch.cat([latents] * 2)
            if do_classifier_free_guidance
            else latents
        )
        # Predict the noise residual
        timestep_tensor = torch.tensor([t], dtype=torch.long, device=device)
        timestep_tensor = timestep_tensor.expand(latent_model_input.shape[0])
        noise_pred = diffusion_model.forward(latent_model_input, timestep_tensor, cond)

        # Perform guidance
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
            )
        # Compute the previous noisy sample x_t -> x_t-1
        latents = ddim_scheduler.step(
            noise_pred, t, latents, **extra_step_kwargs
        ).prev_sample

        yield latents, t


def karra_sample():
    pass
