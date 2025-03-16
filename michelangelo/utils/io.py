# -*- coding: utf-8 -*-

import os
import io
import tarfile
import json
import numpy as np

def mkdir(path):
    """
    Creates a directory at the specified path if it does not already exist.

    Parameters:
    path (str): The path where the directory will be created.

    Returns:
    str: The path of the created directory.
    """
    os.makedirs(path, exist_ok=True)
    return path


def npy_loads(data):
    """
    Loads a numpy array from a byte string.

    Parameters:
    data (bytes): The byte string containing the numpy array data.

    Returns:
    np.ndarray: The loaded numpy array.
    """
    stream = io.BytesIO(data)
    return np.lib.format.read_array(stream)


def npz_loads(data):
    """
    Loads a numpy .npz file from a byte string.

    Parameters:
    data (bytes): The byte string containing the .npz file data.

    Returns:
    np.ndarray: The loaded numpy array.
    """
    return np.load(io.BytesIO(data))


def json_loads(data):
    """
    Parses a JSON string into a Python object.

    Parameters:
    data (str): The JSON string to be parsed.

    Returns:
    object: The parsed Python object.
    """
    return json.loads(data)


def load_json(filepath):
    """
    Loads a JSON file from a file path into a Python object.

    Parameters:
    filepath (str): The file path of the JSON file.

    Returns:
    object: The parsed Python object.
    """
    with open(filepath, "r") as f:
        data = json.load(f)
        return data


def write_json(filepath, data):
    """
    Writes a Python object to a JSON file.

    Parameters:
    filepath (str): The file path where the JSON file will be written.
    data (object): The Python object to be written.
    """
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)


def extract_tar(tar_path, tar_cache_folder):
    """
    Extracts a tar file to a specified folder.

    Parameters:
    tar_path (str): The file path of the tar file.
    tar_cache_folder (str): The folder where the tar file will be extracted.

    Returns:
    list: A list of sorted unique identifiers (UIDs) of the extracted files.
    """
    with tarfile.open(tar_path, "r") as tar:
        tar.extractall(path=tar_cache_folder)

    tar_uids = sorted(os.listdir(tar_cache_folder))
    print(f"extract tar: {tar_path} to {tar_cache_folder}")
    return tar_uids
