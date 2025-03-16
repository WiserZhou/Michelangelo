# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
import PIL.Image
from typing import Optional

import trimesh

def save_obj(pointnp_px3, facenp_fx3, fname):
    """
    Saves a 3D mesh to an OBJ file.

    This function writes the vertices and faces of a 3D mesh to a file in OBJ format. The OBJ format is a simple text-based format for 3D models.

    Parameters:
    - pointnp_px3 (numpy array): A numpy array of shape (n, 3) where n is the number of vertices in the mesh. Each row represents
    a vertex in 3D space.
    - facenp_fx3 (numpy array): A numpy array of shape (m, 3) where m is the number of faces in the mesh. Each row represents a face
    defined by three vertices.
    - fname (str): The file name where the mesh will be saved.

    Returns:
    - None
    """
    # Open the file in write mode
    fid = open(fname, "w")
    # Initialize an empty string to store the content to be written to the file
    write_str = ""
    # Iterate over each vertex in the mesh
    for pidx, p in enumerate(pointnp_px3):
        # Convert the vertex to a string in the format required by the OBJ file
        pp = p
        write_str += "v %f %f %f\n" % (pp[0], pp[1], pp[2])
    
    # Iterate over each face in the mesh
    for i, f in enumerate(facenp_fx3):
        # Adjust the face indices to start from 1 as required by the OBJ format
        f1 = f + 1
        # Convert the face to a string in the format required by the OBJ file
        write_str += "f %d %d %d\n" % (f1[0], f1[1], f1[2])
    
    # Write the content to the file
    fid.write(write_str)
    # Close the file
    fid.close()
    # Return from the function
    return

def savemeshtes2(pointnp_px3, tcoords_px2, facenp_fx3, facetex_fx3, tex_map, fname):
    """
    Saves a 3D mesh with texture coordinates and a material file to an OBJ file.

    This function writes the vertices, texture coordinates, and faces of a 3D mesh to a file in OBJ format. Additionally, it creates a 
    material file (MTL) and a texture image file (PNG) to accompany the mesh.

    Parameters:
    - pointnp_px3 (numpy array): A numpy array of shape (n, 3) where n is the number of vertices in the mesh. Each row represents a vertex 
    in 3D space.
    - tcoords_px2 (numpy array): A numpy array of shape (n, 2) where n is the number of vertices in the mesh. Each row represents a texture coordinate.
    - facenp_fx3 (numpy array): A numpy array of shape (m, 3) where m is the number of faces in the mesh. Each row represents a face defined 
    by three vertices.
    - facetex_fx3 (numpy array): A numpy array of shape (m, 3) where m is the number of faces in the mesh. Each row represents a face defined 
    by three texture coordinates.
    - tex_map (numpy array): A numpy array representing the texture map image.
    - fname (str): The file name where the mesh will be saved.

    Returns:
    - None
    """
    # Split the file name into its directory and base name
    fol, na = os.path.split(fname)
    # Remove the file extension from the base name
    na, _ = os.path.splitext(na)

    # Construct the material file name
    matname = "%s/%s.mtl" % (fol, na)
    # Open the material file in write mode
    fid = open(matname, "w")
    # Write the material properties to the file
    # Define a new material named "material_0"
    fid.write("newmtl material_0\n")
    # Set the diffuse color to white (1 1 1)
    fid.write("Kd 1 1 1\n")
    # Set the ambient color to black (0 0 0), effectively disabling ambient lighting
    fid.write("Ka 0 0 0\n")
    # Set the specular color to a light gray (0.4 0.4 0.4), contributing to the material's shininess
    fid.write("Ks 0.4 0.4 0.4\n")
    # Set the specular exponent to 10, controlling the sharpness of the specular highlight
    fid.write("Ns 10\n")
    # Set the illumination model to 2, indicating a non-physical lighting model with highlights
    fid.write("illum 2\n")
    # Specify the diffuse texture map
    fid.write("map_Kd %s.png\n" % na)
    # Close the material file
    fid.close()
    ####

    # Open the OBJ file in write mode
    fid = open(fname, "w")
    # Reference the material file
    fid.write("mtllib %s.mtl\n" % na)

    # Write the vertices to the file
    for pidx, p in enumerate(pointnp_px3):
        pp = p
        fid.write("v %f %f %f\n" % (pp[0], pp[1], pp[2]))

    # Write the texture coordinates to the file
    for pidx, p in enumerate(tcoords_px2):
        pp = p
        fid.write("vt %f %f\n" % (pp[0], pp[1]))

    # Specify the material to use
    fid.write("usemtl material_0\n")
    # Write the faces to the file, including texture coordinates
    for i, f in enumerate(facenp_fx3):
        f1 = f + 1  # Adjust vertex indices to start from 1 as required by OBJ format
        f2 = facetex_fx3[i] + 1  # Adjust texture coordinate indices to start from 1
        fid.write("f %d/%d %d/%d %d/%d\n" % (f1[0], f2[0], f1[1], f2[1], f1[2], f2[2]))
    # Close the OBJ file
    fid.close()

    # Save the texture map image
    PIL.Image.fromarray(np.ascontiguousarray(tex_map), "RGB").save(
        os.path.join(fol, "%s.png" % na))

    return


class MeshOutput(object):

    def __init__(self,
                 mesh_v: np.ndarray,
                 mesh_f: np.ndarray,
                 vertex_colors: Optional[np.ndarray] = None,
                 uvs: Optional[np.ndarray] = None,
                 mesh_tex_idx: Optional[np.ndarray] = None,
                 tex_map: Optional[np.ndarray] = None):

        self.mesh_v = mesh_v
        self.mesh_f = mesh_f
        self.vertex_colors = vertex_colors
        self.uvs = uvs
        self.mesh_tex_idx = mesh_tex_idx
        self.tex_map = tex_map

    def contain_uv_texture(self):
        return (self.uvs is not None) and (self.mesh_tex_idx is not None) and (self.tex_map is not None)

    def contain_vertex_colors(self):
        return self.vertex_colors is not None

    def export(self, fname):

        if self.contain_uv_texture():
            savemeshtes2(
                self.mesh_v,
                self.uvs,
                self.mesh_f,
                self.mesh_tex_idx,
                self.tex_map,
                fname
            )

        elif self.contain_vertex_colors():
            mesh_obj = trimesh.Trimesh(vertices=self.mesh_v, faces=self.mesh_f, vertex_colors=self.vertex_colors)
            mesh_obj.export(fname)

        else:
            save_obj(
                self.mesh_v,
                self.mesh_f,
                fname
            )



