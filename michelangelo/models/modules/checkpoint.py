# -*- coding: utf-8 -*-
"""
Adapted from: https://github.com/openai/guided-diffusion/blob/22e0df8183507e13a7813f8d38d51b072ca1e67c/guided_diffusion/nn.py#L124
"""

import torch
from typing import Callable, Iterable, Sequence, Union


def checkpoint(
    func: Callable[..., Union[torch.Tensor, Sequence[torch.Tensor]]],
    inputs: Sequence[torch.Tensor],
    params: Iterable[torch.Tensor],
    flag: bool,
    use_deepspeed: bool = False
):
    """
    This function is used to evaluate a given function without caching intermediate activations.
    This allows for reduced memory usage at the expense of extra compute in the backward pass.
    :param func: The function to evaluate.
    :param inputs: The argument sequence to pass to `func`.
    :param params: A sequence of parameters `func` depends on but does not explicitly take as arguments.
    :param flag: If False, gradient checkpointing is disabled.
    :param use_deepspeed: If True, deepspeed is used for checkpointing.
    """
    # Check if gradient checkpointing is enabled
    if flag:
        # If deepspeed is used for checkpointing, import it and use its checkpointing function
        if use_deepspeed:
            import deepspeed
            return deepspeed.checkpointing.checkpoint(func, *inputs)
        # If not using deepspeed, prepare arguments for the custom checkpointing function
        else:
            args = tuple(inputs) + tuple(params)
            # Apply the custom checkpointing function
            return CheckpointFunction.apply(func, len(inputs), *args)
    # If gradient checkpointing is disabled, directly call the function without checkpointing
    else:
        return func(*inputs)


class CheckpointFunction(torch.autograd.Function):
    """
    This class represents a custom autograd function for gradient checkpointing.
    It is used to evaluate a given function without caching intermediate activations.
    This allows for reduced memory usage at the expense of extra compute in the backward pass.
    """
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, run_function, length, *args):
        """
        This method is used to perform the forward pass of the custom autograd function.
        It takes the function to evaluate, the length of the input arguments, and the input arguments.
        It then evaluates the function without caching intermediate activations and returns the output.
        """
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])

        with torch.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, *output_grads):
        """
        This method is used to perform the backward pass of the custom autograd function.
        It takes the output gradients and computes the input gradients.
        It then returns the input gradients.
        """
        ctx.input_tensors = [x.detach().requires_grad_(True) for x in ctx.input_tensors]
        with torch.enable_grad():
            # Fixes a bug where the first op in run_function modifies the
            # Tensor storage in place, which is not allowed for detach()'d
            # Tensors.
            shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
            output_tensors = ctx.run_function(*shallow_copies)
        input_grads = torch.autograd.grad(
            output_tensors,
            ctx.input_tensors + ctx.input_params,
            output_grads,
            allow_unused=True,
        )
        del ctx.input_tensors
        del ctx.input_params
        del output_tensors
        return (None, None) + input_grads
