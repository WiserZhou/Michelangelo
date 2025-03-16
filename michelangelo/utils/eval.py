# -*- coding: utf-8 -*-

import torch


def compute_psnr(x, y, data_range: float = 2, eps: float = 1e-7):
    """
    Computes the Peak Signal-to-Noise Ratio (PSNR) between two tensors x and y.

    This function calculates the PSNR between two tensors x and y, which is a measure of the difference between two images. The PSNR is defined 
    as the ratio of the maximum possible power of a signal to the power of corrupting noise that affects the fidelity of its representation.

    Parameters:
    x (torch.Tensor): The first tensor to compare.
    y (torch.Tensor): The second tensor to compare.
    data_range (float, optional): The maximum possible value of the data. Defaults to 2.
    eps (float, optional): A small value added to the denominator for numerical stability. 
    Defaults to 1e-7.

    Returns:
    torch.Tensor: The computed PSNR value.
    """
    # Calculate the Mean Squared Error (MSE) between x and y
    mse = torch.mean((x - y) ** 2)
    # Calculate the PSNR using the formula: PSNR = 10 * log10(data_range / (mse + eps))
    psnr = 10 * torch.log10(data_range / (mse + eps))

    return psnr
