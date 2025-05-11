import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from xmipp_metadata.metadata import XmippMetaData
from xmipp_metadata.image_handler import ImageHandler
from Volume_projection import VolumeProjection

class MetadataDataset(Dataset):
    def __init__(self, metadata_file_path):
        self.md = XmippMetaData(metadata_file_path) #read metadata table
        self.imgs = self.md.getMetaDataImage(list(range(len(self.md)))).astype(np.float32) #read images
        self.metadata_labels = self.md.getMetaDataLabels() #get the labels of the metadata table
        #self.projection_model = VolumeProjection(volume_path)  # to initialize the projection model
        self.idx=list(range(len(self.md)))

    def __len__(self):
        return len(self.md)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Get image and print it
        image = self.imgs[idx]
        #print("The shape of the image is:", image.shape)    

        all_metadata = {label: self.md[idx, label] for label in self.metadata_labels} #get all mmetadata from the sample and store it 

        euler_angles = np.array(self.md[idx, ["angleRot", "angleTilt", "anglePsi"]], dtype=np.float32)
        # Get Euler angles and convert to a tensor
        #euler_angles_tensor = torch.tensor(euler_angles, dtype=torch.float32)

       # projection = self.projection_model.forward(euler_angles, shift_x, shift_y)
        #subtraction = np.abs(image - np.squeeze(projection.numpy()))

        
        return {
          # 'image': torch.from_numpy(image.astype(np.float32)), #la mayoria de redes trabajam en float 32
          #  'angles': torch.from_numpy(euler_angles.astype(np.float32)), 
          'metadata': all_metadata, 
          'angles': euler_angles,
          'images': image,
          'idx': idx,

        }
    




    

