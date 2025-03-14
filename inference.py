# -*- coding: utf-8 -*-
import os
import time
from collections import OrderedDict
from typing import Optional, List
import argparse
from functools import partial

from einops import repeat, rearrange
import numpy as np
from PIL import Image
import trimesh
import cv2

import torch
import pytorch_lightning as pl

from michelangelo.models.tsal.tsal_base import Latent2MeshOutput
from michelangelo.models.tsal.inference_utils import extract_geometry
from michelangelo.utils.misc import get_config_from_file, instantiate_from_config
from michelangelo.utils.visualizers.pythreejs_viewer import PyThreeJSViewer
from michelangelo.utils.visualizers import html_util

def load_model(args):
    """
    Loads a model from a configuration file and a checkpoint path.
    
    This function first loads the model configuration from a YAML file specified by `args.config_path`.
    If the configuration contains a nested "model" configuration, it extracts that nested configuration.
    Then, it instantiates the model from the configuration, loads the model weights from the checkpoint file specified by `args.ckpt_path`,
    moves the model to a CUDA device (if available), and sets the model to evaluation mode.
    
    Args:
        args: An argparse.Namespace object containing the configuration path and checkpoint path.
        
    Returns:
        model: The loaded model instance.
    """
    # Load the model configuration from the specified file path
    model_config = get_config_from_file(args.config_path)
    # If the configuration contains a nested "model" configuration, extract it
    if hasattr(model_config, "model"):
        model_config = model_config.model

    # Instantiate the model from the configuration and load the model weights from the checkpoint file
    model = instantiate_from_config(model_config, ckpt_path=args.ckpt_path)
    # Move the model to a CUDA device if available
    model = model.cuda()
    # Set the model to evaluation mode
    model = model.eval()

    return model

def load_surface(fp):
    """
    Load and preprocess a point cloud surface from a .npz file.
    
    Args:
        fp: File path to the .npz file containing point cloud data
        
    Returns:
        surface: Tensor of shape (1, N, 6) containing N randomly sampled points
                with their surface normals, moved to GPU
                First 3 dimensions are XYZ coordinates
                Last 3 dimensions are normal vector components
    """
    # Load points and normals from the .npz file
    with np.load(args.pointcloud_path) as input_pc:
        surface = input_pc['points']    # Point coordinates 
        normal = input_pc['normals']    # Surface normal vectors
    
    # Randomly sample 4096 points to get a fixed-size subset
    rng = np.random.default_rng()
    ind = rng.choice(surface.shape[0], 4096, replace=False)
    
    # Convert numpy arrays to PyTorch tensors
    surface = torch.FloatTensor(surface[ind])  # Shape: (4096, 3)
    normal = torch.FloatTensor(normal[ind])    # Shape: (4096, 3)
    
    # Concatenate points and normals, add batch dimension, move to GPU
    surface = torch.cat([surface, normal], dim=-1).unsqueeze(0).cuda()
    
    return surface

def prepare_image(args, number_samples=2):
    # Read the image from the specified file path
    image = cv2.imread(f"{args.image_path}")
    # Convert the image from BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Convert the image to a PyTorch tensor and normalize it
    image_pt = torch.tensor(image).float()
    image_pt = image_pt / 255 * 2 - 1
    # Rearrange the dimensions of the image
    image_pt = rearrange(image_pt, "h w c -> c h w")
    
    # Repeat the image to generate the specified number of samples
    image_pt = repeat(image_pt, "c h w -> b c h w", b=number_samples)

    return image_pt

def save_output(args, mesh_outputs):
    """
    Saves the output meshes to the specified directory.
    
    This function iterates over the list of mesh outputs, reverses the order of the mesh faces to ensure correct orientation,
    and then exports each mesh to an OBJ file in the specified output directory. The file names are numbered sequentially.
    
    Args:
        args: An argparse.Namespace object containing the output directory path.
        mesh_outputs: A list of mesh outputs to be saved.
        
    Returns:
        0: Indicates successful execution.
    """
    # Ensure the output directory exists, creating it if necessary
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Iterate over each mesh output
    for i, mesh in enumerate(mesh_outputs):
        # Reverse the order of the mesh faces to ensure correct orientation
        mesh.mesh_f = mesh.mesh_f[:, ::-1]
        # Create a trimesh object from the mesh vertices and faces
        # mesh_v -> vertices; mesh_f -> faces
        mesh_output = trimesh.Trimesh(mesh.mesh_v, mesh.mesh_f)
        
        # Construct the file name for the current mesh output
        name = str(i) + "_out_mesh.obj"
        # Export the mesh to an OBJ file in the specified output directory, including normals
        mesh_output.export(os.path.join(args.output_dir, name), include_normals=True)
    
    # Print a message indicating the completion of the mesh saving process
    print(f'-----------------------------------------------------------------------------')
    print(f'>>> Finished and mesh saved in {args.output_dir}')
    print(f'-----------------------------------------------------------------------------')        

    return 0

def reconstruction(args, model, bounds=(-1.25, -1.25, -1.25, 1.25, 1.25, 1.25), octree_depth=7, num_chunks=10000):
    """
    This function reconstructs a mesh from a point cloud using the provided model and saves it to a specified directory.
    
    Args:
        args: An argparse.Namespace object containing the path to the point cloud file and the output directory.
        model: The model instance to use for reconstruction.
        bounds: A tuple specifying the bounds of the reconstruction space.
        octree_depth: The depth of the octree to use for reconstruction.
        num_chunks: The number of chunks to divide the reconstruction process into.
        
    Returns:
        0: Indicates successful execution.
    """
    # Load the surface from the point cloud file path
    surface = load_surface(args.pointcloud_path)
    
    # Encoding step: Extract shape embeddings and latents from the surface
    # model.model: CLIPAlignedShapeAsLatentModule
    shape_embed, shape_latents = model.model.encode_shape_embed(surface, return_latents=True)    
    # Apply KL encoding to the shape latents to obtain a probabilistic representation
    # model.model.shape_model: AlignedShapeLatentPerceiver
    shape_zq, posterior = model.model.shape_model.encode_kl_embed(shape_latents)

    # Decoding step: Decode the probabilistic representation back to latents
    # model.model.shape_model: AlignedShapeLatentPerceiver
    latents = model.model.shape_model.decode(shape_zq)
    # Partially apply the query_geometry function with the decoded latents to obtain a geometric function
    geometric_func = partial(model.model.shape_model.query_geometry, latents=latents)
    
    # Reconstruction step: Extract geometry from the geometric function
    mesh_v_f, has_surface = extract_geometry(
        geometric_func=geometric_func,
        device=surface.device,
        batch_size=surface.shape[0],
        bounds=bounds,
        octree_depth=octree_depth,
        num_chunks=num_chunks,
    )
    # Create a trimesh object from the extracted mesh vertices and faces
    recon_mesh = trimesh.Trimesh(mesh_v_f[0][0], mesh_v_f[0][1])
    
    # Save the reconstructed mesh to the specified output directory
    os.makedirs(args.output_dir, exist_ok=True)
    recon_mesh.export(os.path.join(args.output_dir, 'reconstruction.obj'))    
    
    # Print a message indicating the completion of the reconstruction and saving process
    print(f'-----------------------------------------------------------------------------')
    print(f'>>> Finished and mesh saved in {os.path.join(args.output_dir, "reconstruction.obj")}')
    print(f'-----------------------------------------------------------------------------')
    
    return 0

def image2mesh(args, model, guidance_scale=7.5, box_v=1.1, octree_depth=7):
    """
    This function generates a 3D mesh from an input image using the provided model.
    
    Args:
        args (argparse.Namespace): An argparse.Namespace object containing the path to the input image and the output directory.
        model: The model instance to use for generating the mesh.
        guidance_scale (float, optional): The scale factor for guidance in the model. Defaults to 7.5.
        box_v (float, optional): The scale factor for the bounding box of the mesh. Defaults to 1.1.
        octree_depth (int, optional): The depth of the octree to use for mesh generation. Defaults to 7.
    
    Returns:
        int: 0 indicating successful execution.
    """
    # Prepare the input image for the model
    sample_inputs = {
        "image": prepare_image(args)
    }
    
    # Sample the model to generate mesh outputs
    mesh_outputs = model.sample(
        sample_inputs,
        sample_times=1,  # Sample once
        guidance_scale=guidance_scale,  # Apply guidance scale
        return_intermediates=False,  # Do not return intermediate results
        bounds=[-box_v, -box_v, -box_v, box_v, box_v, box_v],  # Define the bounding box for the mesh
        octree_depth=octree_depth,  # Set the octree depth for mesh generation
    )[0]  # Take the first output
    
    # Save the generated mesh output
    save_output(args, mesh_outputs)
    
    return 0

def text2mesh(args, model, num_samples=2, guidance_scale=7.5, box_v=1.1, octree_depth=7):
    """
    This function generates a 3D mesh from an input text description using the provided model.
    
    Args:
        args (argparse.Namespace): An argparse.Namespace object containing the input text and the output directory.
        model: The model instance to use for generating the mesh.
        num_samples (int, optional): The number of samples to generate. Defaults to 2.
        guidance_scale (float, optional): The scale factor for guidance in the model. Defaults to 7.5.
        box_v (float, optional): The scale factor for the bounding box of the mesh. Defaults to 1.1.
        octree_depth (int, optional): The depth of the octree to use for mesh generation. Defaults to 7.
    
    Returns:
        int: 0 indicating successful execution.
    """
    # Prepare the input text for the model
    sample_inputs = {
        "text": [args.text] * num_samples  # Repeat the input text for the specified number of samples
    }
    
    # Sample the model to generate mesh outputs
    mesh_outputs = model.sample(
        sample_inputs,
        sample_times=1,  # Sample once
        guidance_scale=guidance_scale,  # Apply guidance scale
        return_intermediates=False,  # Do not return intermediate results
        bounds=[-box_v, -box_v, -box_v, box_v, box_v, box_v],  # Define the bounding box for the mesh
        octree_depth=octree_depth,  # Set the octree depth for mesh generation
    )[0]  # Take the first output
    
    # Save the generated mesh output
    save_output(args, mesh_outputs)
    
    return 0

task_dick = {
    'reconstruction': reconstruction,
    'image2mesh': image2mesh,
    'text2mesh': text2mesh,
}

if __name__ == "__main__":
    '''
    1. Reconstruct point cloud
    2. Image-conditioned generation
    3. Text-conditioned generation
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, choices=['reconstruction', 'image2mesh', 'text2mesh'], required=True)
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--pointcloud_path", type=str, default='./example_data/surface.npz', help='Path to the input point cloud')
    parser.add_argument("--image_path", type=str, help='Path to the input image')
    parser.add_argument("--text", type=str, help='Input text within a format: A 3D model of motorcar; Porsche 911.')
    parser.add_argument("--output_dir", type=str, default='./output')
    parser.add_argument("-s", "--seed", type=int, default=0)
    args = parser.parse_args()
    
    pl.seed_everything(args.seed)

    print(f'-----------------------------------------------------------------------------')
    print(f'>>> Running {args.task}')
    args.output_dir = os.path.join(args.output_dir, args.task)
    print(f'>>> Output directory: {args.output_dir}')
    print(f'-----------------------------------------------------------------------------')
    
    task_dick[args.task](args, load_model(args))