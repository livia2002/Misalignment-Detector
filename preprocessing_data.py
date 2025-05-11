import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as T
from xmipp_metadata.metadata import XmippMetaData
from xmipp_metadata.image_handler import ImageHandler
from CUSTOM_DATASET import MetadataDataset
from Volume_projection import VolumeProjection
from WrongVolumeProjection import WrongVolumeProjection
from GaussianWrongVolumeProjection import GaussianWrongVolumeProjection
from ctf import computeCTF
import math 
from CorrelationCoeff import CorrelationCoeff


class Preprocesing: 
    def __init__(self,metadata_file_path,volume_path,sr):
        self.md = XmippMetaData(metadata_file_path)
        self.imgs = torch.from_numpy(self.md.getMetaDataImage(list(range(len(self.md))))).type(torch.float32) #read images
        self.metadata_labels = self.md.getMetaDataLabels() #get the labels of the metadata table  
        self.volume_path=volume_path  
        #for rand projection
        self.max_shift_x = np.amax(self.md[:, "shiftX"])
        self.min_shift_x = np.amin(self.md[:, "shiftX"]) 
        self.max_shift_y = np.amax(self.md[:, "shiftY"])
        self.min_shift_y = np.amin(self.md[:, "shiftY"]) 
        #to calculate ctf vlaues that are cte
       # self.cs= torch.from_numpy(self.md[:,"ctfSphericalAberration"]) #spherical aberration
        #self.kv = torch.from_numpy(self.md[:, "ctfVoltage"] )#spherical voltage
        self.sr=sr 

    def process_data(self, metadata,std):
        image = metadata['images']
        angles = metadata['angles']
        shift_x = metadata['metadata']['shiftX'].type(torch.float32)
        shift_y = metadata['metadata']['shiftY'].type(torch.float32)
        defocusU  = metadata['metadata']['ctfDefocusU'].type(torch.float32)
        defocusV  = metadata['metadata']['ctfDefocusV'].type(torch.float32)
        defocusAngle  = metadata['metadata']['ctfDefocusAngle'].type(torch.float32)  
        cs  = metadata['metadata']['ctfSphericalAberration'].type(torch.float32) 
        kv  = metadata['metadata']['ctfVoltage'].type(torch.float32)       
        batch_size=metadata['angles'].shape[0]
        #compute cf
        ctf=computeCTF(defocusU, defocusV, defocusAngle, cs, kv, self.sr, image.shape[1:], batch_size) #[:, None, ...]
        if ctf.dim() == 3:
        # If x is (N, H, W), add a channel dimension
            ctf = ctf.unsqueeze(1)
        elif ctf.dim() == 2:
            # If ctf is (N, D), reshape to (N, 1, H, W)
            ctf = ctf.view(ctf.size(0), 1, int(math.sqrt(ctf.size(1))), int(math.sqrt(ctf.size(1))))
        elif ctf.dim() == 4:
            # If ctf is already (N, C, H, W), ensure C=1
            if ctf.size(1) != 50:
                raise ValueError("Expected input with 1 channel, but got {} channels.".format(ctf.size(1)))
        else:
            raise ValueError("Unsupported input dimensions: {}".format(ctf.shape))
        # Correct alignment projection
        projection_model = VolumeProjection(self.volume_path)

        projection = projection_model.forward(angles, shift_x, shift_y)
        # Apply the CTF to the correct alignment projection
        projection_fft = torch.fft.ifftshift(torch.fft.fft2(projection))  # Compute FFT of the projection
        projection_ctf = (projection_fft.real * ctf + 1.j * projection_fft.imag * ctf).type(torch.complex64) # Apply the CTF
        projection_ctf = torch.fft.ifft2(torch.fft.ifftshift(projection_ctf)).real  # Convert back to real space
        correlation=CorrelationCoeff
        correct_correlation=correlation.pixel_correlation_map(image,projection_ctf)


        #subs_correct = torch.abs(image - torch.squeeze(projection_ctf))
        #labels_correct = torch.ones(subs_correct.shape[0])
        labels_correct=torch.ones(correct_correlation.shape[0])

        # Wrong alignment projection
        #wrong_projection_model = WrongVolumeProjection(self.volume_path, self.max_shift_x,self.min_shift_x,self.max_shift_y,self.min_shift_y,batch_size)
        wrong_projection_model = GaussianWrongVolumeProjection(self.volume_path)
        #wrong_projection = wrong_projection_model.forward()
        wrong_projection = wrong_projection_model.forward(angles,0.0,std, shift_x, shift_y)
        # Apply the CTF to the wrong alignment projection
    
        wrong_projection_fft = torch.fft.ifftshift(torch.fft.fft2(wrong_projection))  # Compute FFT of the wrong projection
        wrong_projection_ctf = (wrong_projection_fft.real * ctf + 1.j * wrong_projection_fft.imag * ctf).type(torch.complex64)  # Apply the CTF
        wrong_projection_ctf = torch.fft.ifft2(torch.fft.ifftshift(wrong_projection_ctf)).real # Convert back to real space
        wrong_correlation=correlation.pixel_correlation_map(image,wrong_projection_ctf)

        import matplotlib.pyplot as plt
        plt.figure()
        plt.imshow(correct_correlation[0, ...].numpy(), cmap="Greys")
        plt.title('correct')
        plt.show()
        plt.figure()
        plt.imshow(wrong_correlation[0, ...].numpy(), cmap="Greys")
        plt.title('wrong')
        plt.show()


       # subs_wrong = torch.abs(image - torch.squeeze(wrong_projection_ctf))
        #labels_wrong = torch.zeros(subs_wrong.shape[0])  # Misaligned -> label 0
        labels_wrong=torch.zeros(wrong_correlation.shape[0])
        """""   
       
        # DEBUG: Check projections match experimental image
        import matplotlib.pyplot as plt
           
        plt.figure()
        plt.subplot(2,4, 1)
        plt.imshow(image[0, ...].numpy(), cmap="Greys")
        plt.title("Original")
        plt.subplot(2,4, 2)
        plt.imshow(projection[0, 0, ...].numpy(), cmap="Greys")
        plt.title("Projection")
        plt.subplot(2,4, 3)
        plt.imshow(projection_ctf[0, 0, ...].numpy(), cmap="Greys")
        plt.title("CTF Projection")
        plt.subplot(2,4, 4)
        plt.imshow(subs_correct[0,...].numpy(), cmap="Greys")
        plt.title("Substraction")
        plt.subplot(2,4, 5)
        plt.imshow(image[0, ...].numpy(), cmap="Greys")
        plt.title("Original")
        plt.subplot(2,4, 6)
        plt.imshow(wrong_projection[0, 0, ...].numpy(), cmap="Greys")
        plt.title("Wrong Projection")
        plt.subplot(2,4, 7)
        plt.imshow(wrong_projection_ctf[0, 0, ...].numpy(), cmap="Greys")
        plt.title("wrong CTF Projection")
        plt.subplot(2,4, 8)
        plt.imshow(subs_wrong[0,...].numpy(), cmap="Greys")
        plt.title("wrong substraction")
        plt.show()

        plt.figure()
        plt.title("Image")
        #plt.subplot(1,2, 1)
        #plt.imshow(image[0, ...].numpy(),cmap="Greys")
        #plt.subplot(1,2, 2)
        plt.hist(image[0, ...].numpy())
        plt.show()

        plt.figure()
        plt.title("Projection")
        #plt.subplot(1,2, 1)
        #plt.imshow(projection[0, 0, ...].numpy(),cmap="Greys")
        #plt.subplot(1,2, 2)
        plt.hist(projection[0, 0, ...].numpy())
        plt.show()

        plt.figure()
        plt.title("Projection CTF")
        #plt.subplot(1,2, 1)
        #plt.imshow(projection_ctf[0, 0, ...].numpy(),cmap="Greys")
        #plt.subplot(1,2, 2)
        plt.hist(projection_ctf[0, 0, ...].numpy())
        plt.show()

        plt.figure()
        plt.title("Wrong Projection")
        plt.imshow(wrong_projection[0, 0, ...].numpy(), cmap="Greys")
        plt.show()

        plt.figure()
        plt.title("wrong projection ctf")
        plt.imshow(wrong_projection_ctf[0, 0, ...].numpy(), cmap="Greys")
        plt.show()

        """""
           

        subs=torch.cat([correct_correlation, wrong_correlation], dim=0)
        labels=torch.cat([labels_correct, labels_wrong], dim=0)
            
        return subs, labels

"""""    
metadata= XmippMetaData('metadata.xmd')
metadata_labels = metadata.getMetaDataLabels()
preprocessor=Preprocesing('metadata.xmd', 'volume.mrc')
subs_tensor, labels_tensor = preprocessor.process_data(metadata)
"""""   