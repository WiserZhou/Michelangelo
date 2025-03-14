# -*- coding: utf-8 -*-

import importlib
from omegaconf import OmegaConf, DictConfig, ListConfig

import torch
import torch.distributed as dist
from typing import Union


def get_config_from_file(config_file: str) -> Union[DictConfig, ListConfig]:
    config_file = OmegaConf.load(config_file)

    if 'base_config' in config_file.keys():
        if config_file['base_config'] == "default_base":
            base_config = OmegaConf.create()
            # base_config = get_default_config()
        elif config_file['base_config'].endswith(".yaml"):
            base_config = get_config_from_file(config_file['base_config'])
        else:
            raise ValueError(f"{config_file} must be `.yaml` file or it contains `base_config` key.")

        config_file = {key: value for key, value in config_file if key != "base_config"}

        return OmegaConf.merge(base_config, config_file)

    return config_file


def get_obj_from_str(string, reload=False):
    """
    Dynamically imports and returns an object from a given string representation.

    This function takes a string representation of a module and class, and returns the class object.
    It also provides an option to reload the module before importing the class.

    Args:
        string (str): A string representation of the module and class, separated by a dot.
        reload (bool, optional): If True, the module will be reloaded before importing the class. Defaults to False.

    Returns:
        object: The class object dynamically imported from the given string representation.
    """
    # Split the input string into module and class names
    module, cls = string.rsplit(".", 1)
    # If reload is True, import the module and reload it
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    # Import the module and return the class object
    return getattr(importlib.import_module(module, package=None), cls)


def get_obj_from_config(config):
    """
    This function takes a configuration dictionary and returns an instance of the class specified in the configuration.

    Args:
        config (dict): A dictionary containing the configuration for the object to be instantiated.

    Raises:
        KeyError: If the configuration does not contain the 'target' key, which is required to specify the class to instantiate.

    Returns:
        instance: An instance of the class specified in the configuration.
    """
    # Check if the 'target' key is present in the configuration, which is required to specify the class to instantiate.
    if "target" not in config:
        raise KeyError("Expected key `target` to instantiate.")

    # Use the 'target' key to dynamically import and get the class from the configuration.
    cls = get_obj_from_str(config["target"])

    # Return the instantiated object.
    return cls


def instantiate_from_config(config, **kwargs):
    """
    Instantiates an object from a given configuration.

    This function takes a configuration dictionary and optional keyword arguments, 
    and returns an instance of the class specified in the configuration.

    Args:
        config (dict): A dictionary containing the configuration for the object to be instantiated.
        **kwargs: Optional keyword arguments to be passed to the class constructor.

    Raises:
        KeyError: If the configuration does not contain the 'target' key, which is required to specify the class to instantiate.

    Returns:
        instance: An instance of the class specified in the configuration.
    """
    # Check if the 'target' key is present in the configuration, which is required to specify the class to instantiate.
    if "target" not in config:
        raise KeyError("Expected key `target` to instantiate.")

    # Use the 'target' key to dynamically import and get the class from the configuration.
    cls = get_obj_from_str(config["target"])

    # Retrieve the 'params' from the configuration, defaulting to an empty dictionary if not present.
    params = config.get("params", dict())

    # Update the keyword arguments with the 'params' from the configuration.
    kwargs.update(params)

    # Instantiate the class using the updated keyword arguments.
    instance = cls(**kwargs)

    # Return the instantiated object.
    return instance


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def all_gather_batch(tensors):
    """
    Performs all_gather operation on the provided tensors.
    """
    # Queue the gathered tensors
    world_size = get_world_size()
    # There is no need for reduction in the single-proc case
    if world_size == 1:
        return tensors
    tensor_list = []
    output_tensor = []
    for tensor in tensors:
        tensor_all = [torch.ones_like(tensor) for _ in range(world_size)]
        dist.all_gather(
            tensor_all,
            tensor,
            async_op=False  # performance opt
        )

        tensor_list.append(tensor_all)

    for tensor_all in tensor_list:
        output_tensor.append(torch.cat(tensor_all, dim=0))
    return output_tensor
