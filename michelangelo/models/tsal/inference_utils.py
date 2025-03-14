# -*- coding: utf-8 -*-

import torch
from tqdm import tqdm
from einops import repeat
import numpy as np
from typing import Callable, Tuple, List, Union, Optional
from skimage import measure

from michelangelo.graphics.primitives import generate_dense_grid_points


@torch.no_grad()
def extract_geometry(geometric_func: Callable,
                    device: torch.device,
                    batch_size: int = 1,
                    bounds: Union[Tuple[float], List[float], float] = (-1.25, -1.25, -1.25, 1.25, 1.25, 1.25),
                    octree_depth: int = 7,
                    num_chunks: int = 10000,
                    disable: bool = True):
    """
    Extracts geometry information from a given geometric function.

    This function is designed to extract geometry information from a given geometric function, which is typically a 
    neural network that predicts occupancy or density values for 3D points. It generates a dense grid of points within 
    a specified bounding box, evaluates the geometric function at these points, and then uses the marching cubes algorithm
    to extract a mesh from the predicted values.

    Args:
        geometric_func (Callable): The geometric function to evaluate. This function should take a tensor of 3D points as 
        input and return a tensor of occupancy or density values.
        device (torch.device): The device on which to perform the computations.
        batch_size (int, optional): The batch size to use for evaluating the geometric function. Defaults to 1.
        bounds (Union[Tuple[float], List[float], float], optional): The bounds of the region to extract geometry from. If a 
        single float is provided, it is used as the bounds for all dimensions. Defaults to (-1.25, -1.25, -1.25, 1.25, 1.25, 1.25).
        octree_depth (int, optional): The depth of the octree to use for generating the dense grid of points. Defaults to 7.
        num_chunks (int, optional): The number of chunks to divide the dense grid into for processing. Defaults to 10000.
        disable (bool, optional): A flag to disable the tqdm progress bar. Defaults to True.

    Returns:
        Tuple[List[Tuple[np.ndarray, np.ndarray]], np.ndarray]: A tuple containing a list of tuples, where each tuple contains 
        the vertices and faces of a mesh, and a boolean array indicating whether a surface was found for each batch element.
    """

    # If the bounds are a single float, convert it to a list of bounds for all dimensions
    if isinstance(bounds, float):
        bounds = [-bounds, -bounds, -bounds, bounds, bounds, bounds]

    # Extract the minimum, maximum, and size of the bounding box
    bbox_min = np.array(bounds[0:3])
    bbox_max = np.array(bounds[3:6])
    bbox_size = bbox_max - bbox_min

    # Generate a dense grid of points within the bounding box
    xyz_samples, grid_size, length = generate_dense_grid_points(
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        octree_depth=octree_depth,
        indexing="ij"
    )
    # Convert the numpy array of dense grid points to a PyTorch FloatTensor for further processing.
    xyz_samples = torch.FloatTensor(xyz_samples)

    # Initialize an empty list to store the logits for each batch
    batch_logits = []
    # Iterate over the dense grid of points in chunks
    for start in tqdm(range(0, xyz_samples.shape[0], num_chunks),
                    desc="Implicit Function:", disable=disable, leave=False):
        # Extract the queries for the current chunk
        queries = xyz_samples[start: start + num_chunks, :].to(device)
        # Repeat the queries for each batch element
        batch_queries = repeat(queries, "p c -> b p c", b=batch_size)
        # decode for the batch queries
        logits = geometric_func(batch_queries)
        # Append the logits to the list of batch logits
        batch_logits.append(logits.cpu())

    # Concatenate the batch logits and reshape them into a grid
    grid_logits = torch.cat(batch_logits, dim=1).view((batch_size, grid_size[0], grid_size[1], grid_size[2])).numpy()

    # Initialize an empty list to store the vertices and faces of the mesh
    mesh_v_f = []
    # Initialize a boolean array to indicate whether a surface was found for each batch element
    has_surface = np.zeros((batch_size,), dtype=np.bool_)
    # Iterate over each batch element
    for i in range(batch_size):
        try:
            # Use the marching cubes algorithm to extract the mesh from the grid logits
            vertices, faces, normals, _ = measure.marching_cubes(grid_logits[i], 0, method="lewiner")
            # Rescale the vertices to the original bounding box
            vertices = vertices / grid_size * bbox_size + bbox_min
            # Append the vertices and faces to the list of mesh vertices and faces
            mesh_v_f.append((vertices.astype(np.float32), np.ascontiguousarray(faces)))
            # Set the surface flag to True
            has_surface[i] = True

        except ValueError:
            # If the marching cubes algorithm fails, append None to the list of mesh vertices and faces
            mesh_v_f.append((None, None))
            # Set the surface flag to False
            has_surface[i] = False

        except RuntimeError:
            # If the marching cubes algorithm fails, append None to the list of mesh vertices and faces
            mesh_v_f.append((None, None))
            # Set the surface flag to False
            has_surface[i] = False

    # Return the list of mesh vertices and faces and the boolean array indicating whether a surface was found for each batch element
    return mesh_v_f, has_surface
