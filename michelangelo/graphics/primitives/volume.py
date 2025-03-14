# -*- coding: utf-8 -*-

import numpy as np

def generate_dense_grid_points(bbox_min: np.ndarray,
                            bbox_max: np.ndarray,
                            octree_depth: int,
                            indexing: str = "ij"):
    """
    Generates dense grid points within a bounding box based on the specified octree depth.

    Parameters:
    bbox_min (np.ndarray): The minimum coordinates of the bounding box.
    bbox_max (np.ndarray): The maximum coordinates of the bounding box.
    octree_depth (int): The depth of the octree, which determines the density of the grid points.
    indexing (str, optional): The indexing order for the meshgrid. Defaults to "ij".

    Returns:
    tuple: A tuple containing the dense grid points (xyz), the size of the grid (grid_size), and the length of the bounding box (length).
    """
    # Calculate the length of the bounding box along each dimension
    length = bbox_max - bbox_min
    # Calculate the number of cells based on the octree depth
    # 2^(octree_depth)
    num_cells = np.exp2(octree_depth)
    # Generate evenly spaced values along each dimension
    x = np.linspace(bbox_min[0], bbox_max[0], int(num_cells) + 1, dtype=np.float32)
    y = np.linspace(bbox_min[1], bbox_max[1], int(num_cells) + 1, dtype=np.float32)
    z = np.linspace(bbox_min[2], bbox_max[2], int(num_cells) + 1, dtype=np.float32)
    # Create a 3D meshgrid based on the generated values
    [xs, ys, zs] = np.meshgrid(x, y, z, indexing=indexing)
    # Stack the meshgrid arrays along the last axis to form a 3D array of points
    xyz = np.stack((xs, ys, zs), axis=-1)
    # Reshape the 3D array into a 2D array of points
    xyz = xyz.reshape(-1, 3)
    # Calculate the size of the grid based on the number of cells
    grid_size = [int(num_cells) + 1, int(num_cells) + 1, int(num_cells) + 1]

    return xyz, grid_size, length

