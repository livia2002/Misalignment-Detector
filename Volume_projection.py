import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from xmipp_metadata.metadata import XmippMetaData
from xmipp_metadata.image_handler import ImageHandler

class VolumeProjection:
    def __init__(self, volume_file_path):
        #the input_dim defines the dmensions of the volume and the it stores it
        self.volume = torch.from_numpy(ImageHandler(volume_file_path).getData().astype(np.float32)) #read volume
        self.input_dim = self.volume.shape[0]  # The size of the 3D volume

        # Combined rotation matrix, therefore multiply the individual rotation matrices in that sequence, starting with R_z2, followed by R_y, and then R_z1
    def rotate_volume(self, volume, euler_angles):
        
       # Rotate the 3D volume using the given Euler angles (in degrees).
        
        rot, tilt, psi = torch.deg2rad(euler_angles).unbind(dim=1)  #convert the euler angles to radians
        rot, tilt, psi = -rot, -tilt, -psi

        # Define rotation matrices for entire batch
        batch_size = euler_angles.shape[0]
        # z1 is a rotation around the z-axis
        R_z1 = torch.stack([
            torch.stack([torch.cos(rot), -torch.sin(rot), torch.zeros_like(rot), torch.zeros_like(rot)], dim=1),
            torch.stack([torch.sin(rot), torch.cos(rot), torch.zeros_like(rot), torch.zeros_like(rot)], dim=1),
            torch.stack([torch.zeros_like(rot), torch.zeros_like(rot), torch.ones_like(rot), torch.zeros_like(rot)], dim=1),
            torch.stack([torch.zeros_like(rot), torch.zeros_like(rot), torch.zeros_like(rot), torch.ones_like(rot)], dim=1),
        ], dim=2)  # Shape: (batch_size, 4, 4)
    #y is a rotation around the y-axis.
        R_y = torch.stack([
            torch.stack([torch.cos(tilt), torch.zeros_like(tilt), torch.sin(tilt), torch.zeros_like(tilt)], dim=1),
            torch.stack([torch.zeros_like(tilt), torch.ones_like(tilt), torch.zeros_like(tilt), torch.zeros_like(tilt)], dim=1),
            torch.stack([-torch.sin(tilt), torch.zeros_like(tilt), torch.cos(tilt), torch.zeros_like(tilt)], dim=1),
            torch.stack([torch.zeros_like(tilt), torch.zeros_like(tilt), torch.zeros_like(tilt), torch.ones_like(tilt)], dim=1),
        ], dim=2)  # Shape: (batch_size, 4, 4)
    # z2 is a rotation around the z-axis
        R_z2 = torch.stack([
            torch.stack([torch.cos(psi), -torch.sin(psi), torch.zeros_like(psi), torch.zeros_like(psi)], dim=1),
            torch.stack([torch.sin(psi), torch.cos(psi), torch.zeros_like(psi), torch.zeros_like(psi)], dim=1),
            torch.stack([torch.zeros_like(psi), torch.zeros_like(psi), torch.ones_like(psi), torch.zeros_like(psi)], dim=1),
            torch.stack([torch.zeros_like(psi), torch.zeros_like(psi), torch.zeros_like(psi), torch.ones_like(psi)], dim=1),
        ], dim=2)  # Shape: (batch_size, 4, 4) al ò

        # R_temp = torch.bmm(R_z1.transpose(1, 2), R_y.transpose(1, 2))  # Shape: (batch_size, 4, 4)
        # R = torch.bmm(R_temp, R_z2.transpose(1, 2))  # Shape: (batch_size, 4, 4)
        R = R_z1 @ R_y @ R_z2

        # Extract the top-left 3x4 submatrix for affine_grid
        R_affine = R[:, :-1, :]  # Shape: (batch_size, 3, 4)

        # Create a grid for rotation
        grid = F.affine_grid(R_affine, [batch_size, 1, self.input_dim, self.input_dim, self.input_dim], align_corners=False)  # Shape: (batch_size, D, H, W, 3)

        # Rotate the volume using grid_sample
        rotated_volume = F.grid_sample(volume, grid, align_corners=False)  # Shape: (batch_size, channels, D, H, W)
            # Debug: Check the shape of the rotated volume

        return rotated_volume

    def project_volume(self, volume):
    
       # Project the 3D volume into a 2D plane by adding along the z-axis.
        return torch.sum(volume, dim=2) #la 0 es batch size, la 1 es channels y la 2 ya es z

    def shift_projection(self, projection, shift_x, shift_y):
  
       # Shift the 2D projection by the given x and y offsets.
        batch_size=projection.shape[0]
        #descomentar esta linea cuando tengamos batch size
       # batch_size, height, width = projection.shape 
       #cuando los shift sean una matriz, hacer una matriz en vez de repeat y añadir los shifts
        half_box = 1. / (0.5 * projection.shape[-1])
        grid = F.affine_grid(
            torch.stack([
                torch.stack([torch.ones_like(shift_x), torch.zeros_like(shift_x), half_box * shift_x], dim=1),
                torch.stack([torch.zeros_like(shift_y), torch.ones_like(shift_y), half_box * shift_y], dim=1),
            ], dim=1),  # Shape: (batch_size, 2, 3)
            [batch_size, 1, self.input_dim, self.input_dim],
            align_corners=False
        )
        #probar con menos shift por si acaso
        shifted_projection = F.grid_sample(projection, grid, align_corners=False)
        return shifted_projection

    def forward(self, angles, shift_x, shift_y):
        batch_size = angles.shape[0]
        volume = self.volume[None, None, ...].expand(batch_size, -1, -1, -1, -1)
       #rotate, project, and shift the volume.
   
        rotated_volume = self.rotate_volume(volume, angles)
        projection = self.project_volume(rotated_volume)
        shifted_projection = self.shift_projection(projection, shift_x, shift_y)
        return shifted_projection

#how to use it is in CUSTOM DATASET