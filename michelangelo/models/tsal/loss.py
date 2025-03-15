# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Tuple, Dict

from michelangelo.models.modules.distributions import DiagonalGaussianDistribution
from michelangelo.utils.eval import compute_psnr
from michelangelo.utils import misc

class KLNearFar(nn.Module):
    """
    This class defines a loss function for training a model that predicts occupancy and uncertainty.
    It combines binary cross-entropy loss for occupancy prediction with KL divergence loss for uncertainty estimation.
    The loss is weighted for near and far points separately.
    """
    def __init__(self,
                near_weight: float = 0.1,
                kl_weight: float = 1.0,
                num_near_samples: Optional[int] = None):

        """
        Initializes the loss function with weights for near points and KL divergence, and the number of near samples.

        Args:
            near_weight (float): Weight for the binary cross-entropy loss of near points.
            kl_weight (float): Weight for the KL divergence loss.
            num_near_samples (Optional[int]): Number of near samples. If None, it is set to half of the logits shape.
        """
        super().__init__()

        self.near_weight = near_weight
        self.kl_weight = kl_weight
        self.num_near_samples = num_near_samples
        self.geo_criterion = nn.BCEWithLogitsLoss()  # Binary cross-entropy loss for occupancy prediction

    def forward(self,
                posteriors: Optional[DiagonalGaussianDistribution],
                logits: torch.FloatTensor,
                labels: torch.FloatTensor,
                split: Optional[str] = "train", **kwargs) -> Tuple[torch.FloatTensor, Dict[str, float]]:
        """
        Computes the loss and logs for the given inputs.

        Args:
            posteriors (Optional[DiagonalGaussianDistribution]): Posterior distributions for uncertainty estimation.
            logits (torch.FloatTensor): Logits for occupancy prediction.
            labels (torch.FloatTensor): Ground truth labels for occupancy.
            split (Optional[str]): Split of the data (train, val, test).
            **kwargs: Additional keyword arguments.

        Returns:
            loss (torch.Tensor): Total loss.
            log (Dict[str, float]): Dictionary of logs including loss components and metrics.
        """

        # If the number of near samples is not specified, use half of the logits shape
        if self.num_near_samples is None:
            num_vol = logits.shape[1] // 2
        # Otherwise, use the difference between the logits shape and the number of near samples
        else:
            num_vol = logits.shape[1] - self.num_near_samples

        # Split the logits and labels into volume and near parts
        vol_logits = logits[:, 0:num_vol]
        vol_labels = labels[:, 0:num_vol]

        near_logits = logits[:, num_vol:]
        near_labels = labels[:, num_vol:]

        # Calculate the occupancy loss for volume and near parts
        vol_bce = self.geo_criterion(vol_logits.float(), vol_labels.float())
        near_bce = self.geo_criterion(near_logits.float(), near_labels.float())

        # calculate the difference between posterior distribution and ground truth distribution.
        # If there are no posteriors, set the KL loss to 0
        if posteriors is None:
            kl_loss = torch.tensor(0.0, dtype=vol_logits.dtype, device=vol_logits.device)
        # Otherwise, calculate the KL loss
        else:
            kl_loss = posteriors.kl(dims=(1, 2))
            kl_loss = torch.mean(kl_loss)

        # Calculate the total loss
        loss = vol_bce + near_bce * self.near_weight + kl_loss * self.kl_weight

        # Calculate the accuracy and positive ratio
        with torch.no_grad():
            preds = logits >= 0
            accuracy = (preds == labels).float()
            accuracy = accuracy.mean()
            pos_ratio = torch.mean(labels)

        # Create a log dictionary
        log = {
            "{}/total_loss".format(split): loss.clone().detach(),
            "{}/near".format(split): near_bce.detach(),
            "{}/far".format(split): vol_bce.detach(),
            "{}/kl".format(split): kl_loss.detach(),
            "{}/accuracy".format(split): accuracy,
            "{}/pos_ratio".format(split): pos_ratio
        }

        # If there are posteriors, add the mean, std_mean, and std_max to the log dictionary
        if posteriors is not None:
            log[f"{split}/mean"] = posteriors.mean.mean().detach()
            log[f"{split}/std_mean"] = posteriors.std.mean().detach()
            log[f"{split}/std_max"] = posteriors.std.max().detach()

        # Return the total loss and the log dictionary
        return loss, log


class KLNearFarColor(nn.Module):
    """
    This class defines a loss function that combines occupancy loss for both near and far points, 
    surface color loss, and KL divergence loss for posteriors. It is designed for tasks that require 
    both geometric and color information, such as scene reconstruction or image synthesis.
    """
    def __init__(self,
                near_weight: float = 0.1,
                kl_weight: float = 1.0,
                color_weight: float = 1.0,
                color_criterion: str = "mse",
                num_near_samples: Optional[int] = None):

        """
        Initializes the loss function with specified weights for near points, KL divergence, and color loss.
        It also sets the criterion for color loss based on the input.

        Args:
            near_weight (float, optional): Weight for the loss of near points. Defaults to 0.1.
            kl_weight (float, optional): Weight for the KL divergence loss. Defaults to 1.0.
            color_weight (float, optional): Weight for the color loss. Defaults to 1.0.
            color_criterion (str, optional): Criterion for color loss, either 'mse' or 'l1'. Defaults to 'mse'.
            num_near_samples (Optional[int], optional): Number of near samples. Defaults to None.
        """
        super().__init__()

        self.color_weight = color_weight
        self.near_weight = near_weight
        self.kl_weight = kl_weight
        self.num_near_samples = num_near_samples

        # Sets the color loss criterion based on the input
        if color_criterion == "mse":
            self.color_criterion = nn.MSELoss()

        elif color_criterion == "l1":
            self.color_criterion = nn.L1Loss()

        else:
            raise ValueError(f"{color_criterion} must be [`mse`, `l1`].")

        # Initializes the loss criterion for occupancy (geometry)
        self.geo_criterion = nn.BCEWithLogitsLoss()

    def forward(self,
                posteriors: Optional[DiagonalGaussianDistribution],
                logits: torch.FloatTensor,
                labels: torch.FloatTensor,
                pred_colors: torch.FloatTensor,
                gt_colors: torch.FloatTensor,
                split: Optional[str] = "train", **kwargs) -> Tuple[torch.FloatTensor, Dict[str, float]]:

        """
        Computes the total loss by combining occupancy loss for both near and far points, 
        surface color loss, and KL divergence loss for posteriors.

        Args:
            posteriors (DiagonalGaussianDistribution or torch.distributions.Normal):
            logits (torch.FloatTensor): [B, 2*N], logits[:, 0:N] is the volume points; logits[:, N:2N] is the near points;
            labels (torch.FloatTensor): [B, 2*N], labels[:, 0:N] is the volume points; labels[:, N:2N] is the near points;
            pred_colors (torch.FloatTensor): [B, M, 3]
            gt_colors (torch.FloatTensor): [B, M, 3]
            split (str):
            **kwargs:

        Returns:
            loss (torch.Tensor): The total loss.
            log (dict): A dictionary containing the total loss, near loss, far loss, color loss, KL loss, PSNR, and accuracy.
        """

        if self.num_near_samples is None:
            num_vol = logits.shape[1] // 2
        else:
            num_vol = logits.shape[1] - self.num_near_samples

        vol_logits = logits[:, 0:num_vol]
        vol_labels = labels[:, 0:num_vol]

        near_logits = logits[:, num_vol:]
        near_labels = labels[:, num_vol:]

        # Computes the occupancy loss for both volume and near points
        vol_bce = self.geo_criterion(vol_logits.float(), vol_labels.float())
        near_bce = self.geo_criterion(near_logits.float(), near_labels.float())

        # Computes the surface color loss
        color = self.color_criterion(pred_colors, gt_colors)

        # Computes the KL divergence loss if posteriors are provided
        if posteriors is None:
            kl_loss = torch.tensor(0.0, dtype=pred_colors.dtype, device=pred_colors.device)
        else:
            kl_loss = posteriors.kl(dims=(1, 2))
            kl_loss = torch.mean(kl_loss)

        # Computes the total loss by combining all losses
        loss = vol_bce + near_bce * self.near_weight + color * self.color_weight + kl_loss * self.kl_weight

        # Computes the accuracy and PSNR for logging
        with torch.no_grad():
            preds = logits >= 0
            accuracy = (preds == labels).float()
            accuracy = accuracy.mean()
            psnr = compute_psnr(pred_colors, gt_colors)

        # Prepares the log dictionary
        log = {
            "{}/total_loss".format(split): loss.clone().detach(),
            "{}/near".format(split): near_bce.detach(),
            "{}/far".format(split): vol_bce.detach(),
            "{}/color".format(split): color.detach(),
            "{}/kl".format(split): kl_loss.detach(),
            "{}/psnr".format(split): psnr.detach(),
            "{}/accuracy".format(split): accuracy
        }

        return loss, log


class ContrastKLNearFar(nn.Module):
    """
    This module computes the contrastive loss, KL divergence loss, and occupancy loss for both near and far points.
    It is designed to handle the contrastive learning task for shape-text-image embeddings and the occupancy 
    prediction task for shape reconstruction.
    """
    def __init__(self,
                contrast_weight: float = 1.0,
                near_weight: float = 0.1,
                kl_weight: float = 1.0,
                num_near_samples: Optional[int] = None):

        """
        Initializes the module with weights for contrastive loss, near occupancy loss, KL divergence loss, and the 
        number of near samples.
        
        Args:
            contrast_weight (float, optional): The weight for the contrastive loss. Defaults to 1.0.
            near_weight (float, optional): The weight for the near occupancy loss. Defaults to 0.1.
            kl_weight (float, optional): The weight for the KL divergence loss. Defaults to 1.0.
            num_near_samples (Optional[int], optional): The number of near samples to consider. Defaults to None.
        """
        super().__init__()

        self.labels = None  # Placeholder for labels
        self.last_local_batch_size = None  # Placeholder for the last local batch size

        self.contrast_weight = contrast_weight  # Weight for contrastive loss
        self.near_weight = near_weight  # Weight for near occupancy loss
        self.kl_weight = kl_weight  # Weight for KL divergence loss
        self.num_near_samples = num_near_samples  # Number of near samples
        self.geo_criterion = nn.BCEWithLogitsLoss()  # Criterion for occupancy loss

    def forward(self,
                shape_embed: torch.FloatTensor,
                text_embed: torch.FloatTensor,
                image_embed: torch.FloatTensor,
                logit_scale: torch.FloatTensor,
                posteriors: Optional[DiagonalGaussianDistribution],
                shape_logits: torch.FloatTensor,
                shape_labels: torch.FloatTensor,
                split: Optional[str] = "train", **kwargs):

        """
        Computes the forward pass for the module.
        
        Args:
            shape_embed (torch.FloatTensor): The embedding for shapes.
            text_embed (torch.FloatTensor): The embedding for texts.
            image_embed (torch.FloatTensor): The embedding for images.
            logit_scale (torch.FloatTensor): The scale for logits.
            posteriors (Optional[DiagonalGaussianDistribution], optional): The posteriors for KL divergence loss. 
            Defaults to None.
            shape_logits (torch.FloatTensor): The logits for shape reconstruction.
            shape_labels (torch.FloatTensor): The labels for shape reconstruction.
            split (Optional[str], optional): The split for logging. Defaults to "train".
        """
        local_batch_size = shape_embed.size(0)  # Get the local batch size

        if local_batch_size != self.last_local_batch_size:
            # If the local batch size has changed, update the labels and last local batch size
            self.labels = local_batch_size * misc.get_rank() + torch.arange(
                local_batch_size, device=shape_embed.device
            ).long()
            self.last_local_batch_size = local_batch_size

        # Normalize the embeddings
        shape_embed = F.normalize(shape_embed, dim=-1, p=2)
        text_embed = F.normalize(text_embed, dim=-1, p=2)
        image_embed = F.normalize(image_embed, dim=-1, p=2)

        # Gather features from all GPUs
        shape_embed_all, text_embed_all, image_embed_all = misc.all_gather_batch(
            [shape_embed, text_embed, image_embed]
        )

        # Compute cosine similarity as logits for contrastive loss
        logits_per_shape_text = logit_scale * shape_embed @ text_embed_all.t()
        logits_per_text_shape = logit_scale * text_embed @ shape_embed_all.t()
        logits_per_shape_image = logit_scale * shape_embed @ image_embed_all.t()
        logits_per_image_shape = logit_scale * image_embed @ shape_embed_all.t()
        contrast_loss = (F.cross_entropy(logits_per_shape_text, self.labels) +
                        F.cross_entropy(logits_per_text_shape, self.labels)) / 2 + \
                        (F.cross_entropy(logits_per_shape_image, self.labels) +
                        F.cross_entropy(logits_per_image_shape, self.labels)) / 2

        # Compute occupancy loss for shape reconstruction
        if self.num_near_samples is None:
            num_vol = shape_logits.shape[1] // 2
        else:
            num_vol = shape_logits.shape[1] - self.num_near_samples

        vol_logits = shape_logits[:, 0:num_vol]
        vol_labels = shape_labels[:, 0:num_vol]

        near_logits = shape_logits[:, num_vol:]
        near_labels = shape_labels[:, num_vol:]

        vol_bce = self.geo_criterion(vol_logits.float(), vol_labels.float())  # Volume occupancy loss
        near_bce = self.geo_criterion(near_logits.float(), near_labels.float())  # Near occupancy loss

        if posteriors is None:
            kl_loss = torch.tensor(0.0, dtype=vol_logits.dtype, device=vol_logits.device)  # KL divergence loss
        else:
            kl_loss = posteriors.kl(dims=(1, 2))
            kl_loss = torch.mean(kl_loss)

        # Compute the total loss
        loss = vol_bce + near_bce * self.near_weight + kl_loss * self.kl_weight + contrast_loss * self.contrast_weight

        # Compute accuracy and other metrics for logging
        with torch.no_grad():
            pred = torch.argmax(logits_per_shape_text, dim=-1)
            correct = pred.eq(self.labels).sum()
            shape_text_acc = 100 * correct / local_batch_size

            pred = torch.argmax(logits_per_shape_image, dim=-1)
            correct = pred.eq(self.labels).sum()
            shape_image_acc = 100 * correct / local_batch_size

            preds = shape_logits >= 0
            accuracy = (preds == shape_labels).float()
            accuracy = accuracy.mean()

            log = {
                "{}/contrast".format(split): contrast_loss.clone().detach(),
                "{}/near".format(split): near_bce.detach(),
                "{}/far".format(split): vol_bce.detach(),
                "{}/kl".format(split): kl_loss.detach(),
                "{}/shape_text_acc".format(split): shape_text_acc,
                "{}/shape_image_acc".format(split): shape_image_acc,
                "{}/total_loss".format(split): loss.clone().detach(),
                "{}/accuracy".format(split): accuracy,
            }

            if posteriors is not None:
                log[f"{split}/mean"] = posteriors.mean.mean().detach()
                log[f"{split}/std_mean"] = posteriors.std.mean().detach()
                log[f"{split}/std_max"] = posteriors.std.max().detach()

        return loss, log
