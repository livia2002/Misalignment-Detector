import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as T
from xmipp_metadata.metadata import XmippMetaData
from xmipp_metadata.image_handler import ImageHandler
from CUSTOM_DATASET import MetadataDataset
from Volume_projection import VolumeProjection
from preprocessing_data import Preprocesing
from WrongVolumeProjection import WrongVolumeProjection
import math 
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from torch.utils.data import Subset
from VGG16 import VGG16

def test_predict(test_model, test_loader, metadata_path, volume_path, sr,std, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_model = test_model.to(device)

    test_model.eval()
    criterion=nn.BCELoss()
    preprocessor = Preprocesing(metadata_path, volume_path, sr)
    #tracking metrics
    total_test_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    all_targets = []
    all_predictions = []

    with torch.no_grad():
        for batch_data in test_loader:
            # Process data
            subs_tensor, labels_tensor = preprocessor.process_data(batch_data,std)
            
            # Move data to device
            inputs = subs_tensor.to(device)
            targets = labels_tensor.to(device)
            
            # Forward pass
            outputs = test_model(inputs).squeeze()
            outputs = torch.sigmoid(outputs)
            
            # Compute loss
            loss = criterion(outputs, targets)
            total_test_loss += loss.item()
            
            # Compute accuracy
            predictions = (outputs > 0.5).float()
            correct_predictions += (predictions == targets).float().sum().item()
            total_samples += targets.size(0)
            
            # Store results (convert to CPU first)
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    avg_test_loss = total_test_loss / len(test_loader)
    test_accuracy = correct_predictions / total_samples * 100

    cm = confusion_matrix(all_targets, all_predictions)
    cm_percentage = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_percentage, annot=True, fmt=".2f", cmap="Blues", 
                xticklabels=["Negative", "Positive"], 
                yticklabels=["Negative", "Positive"])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Final Test Set Confusion Matrix")
    
    # Save the figure
    conf_mat_filename = "final_test_conf_matrix.png"
    plt.savefig(conf_mat_filename, bbox_inches='tight', dpi=300)
    plt.close()

    #print results
    print(f"Final Test Results:")
    print(f"Test Loss: {avg_test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.2f}%")
    
    return {
        'loss': avg_test_loss,
        'accuracy': test_accuracy,
        'confusion_matrix': cm,
        'confusion_matrix_percentage': cm_percentage
    }