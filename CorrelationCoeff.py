import numpy as np 
import torch

class CorrelationCoeff: 
    def pixel_correlation_map(image, projection):
        # Load images
        img1 = image
        img2 = torch.squeeze(projection)
        
        # Ensure images are the same size
        if img1.shape != img2.shape:
            raise ValueError("Images must have the same dimensions")
        
        # Calculate means
        x_mean = torch.mean(img1)
        y_mean = torch.mean(img2)
        
        # Calculate Pearsons Correlation Coeff 
        numerator = (img1 - x_mean) * (img2 - y_mean)
        denominator = torch.sqrt(torch.sum((img1 - x_mean)**2) * torch.sum((img2 - y_mean)**2))
        correlation_map = numerator / denominator
        #print(correlation_map)
        return correlation_map

