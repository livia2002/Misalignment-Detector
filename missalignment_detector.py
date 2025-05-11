import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
import torchvision.transforms as T
from CUSTOM_DATASET import MetadataDataset
from preprocessing_data import Preprocesing
import math 
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from torch.utils.data import Subset
from VGG16 import VGG16
from TEST_PREDICT import test_predict

# Hyperparameters
#LEARNING_RATE = 1e-4
#EPOCHS = 10
#BATCH_SIZE= 32 
#  #since from every image we will create 2 projections, we will then half the batch isze

def mrcs_dataloader(metadata_path, batch_size, train_split):
    # Load dataset
    #dataset = np.load(file_path)  # Assuming shape (n_samples, height, width)
    #with this dataset we are loading the metadata for 
    full_dataset = MetadataDataset(metadata_path) #50000
    #train dataset
    train_indices = np.random.choice(len(full_dataset), 100, replace=False)
    train_dataset = Subset(full_dataset, train_indices)
    #this returns  'angles' (euler angles) and 'metadata'

    #test dataset (remaining images)
    all_indices=set(range(len(full_dataset)))
    test_indices= list(all_indices-set(train_indices))
    test_dataset=Subset(full_dataset, test_indices)
    
    # Split dataset for training 
    train_size = int(train_split * len(train_dataset))
    test_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, test_size])
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, int(batch_size/2), shuffle=True) #batch 1/2 length
    val_loader = DataLoader(val_dataset, int(batch_size/2), shuffle=True)
    test_loader = DataLoader(test_dataset, int(batch_size*2), shuffle=True)

        
    return train_loader, val_loader, test_loader #loading takes 20 seconds

class BottleneckBlock(nn.Module):
    expansion = 4  # Increases channel depth

    def __init__(self, in_channels, out_channels, stride=1, downsample=True): 
        super(BottleneckBlock, self).__init__()
        
        # Reduced channels for the 3x3 conv
        #print(out_channels)
        reduced_channels = out_channels // self.expansion
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels, reduced_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(reduced_channels)
        
        self.conv2 = nn.Conv2d(reduced_channels, reduced_channels, kernel_size=3, 
                                stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(reduced_channels)
        
        self.conv3 = nn.Conv2d(reduced_channels, out_channels, kernel_size=1, stride=stride, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        self.relu = nn.ReLU(inplace=True)
        
        # Downsampling layer for matching dimensions
        #self.downsample = downsample
        #self.downsample = downsample or nn.Sequential(
        #    nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
        #    nn.BatchNorm2d(out_channels)
        #)
        #self.downsample = T.Resize(out_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        # First 1x1 convolution
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        # 3x3 convolution
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        # Final 1x1 convolution
        out = self.conv3(out)
        out = self.bn3(out)


        # Handle downsampling if needed
        if self.downsample:
            # identity = self.downsample(x)
            identity = T.Resize(out.shape[-1])(x)
        


        # Residual connection
        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1):
        super(ResNet, self).__init__()
        self.in_channels = 64

        # Initial convolution
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=1, stride=1, padding="same", bias=False)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=1, stride=1, padding="same", bias=False)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding="same", bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)

        # Residual layers
        self.layer1 = self._make_layer(block, self.in_channels, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, 512, layers[3], stride=2)

        # Classification head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 1)  
       # self.fc = nn.Sequential(
        #    nn.Linear(512, 1),
     #  nn.Sigmoid()
    #    )

        # Weight initialization
        self._initialize_weights()

    def _make_layer(self, block, in_channels, out_channels, blocks, stride=1):

        downsample = False
        if stride != 1: #or self.in_channels != out_channels * block.expansion:
            downsample = True
        #    downsample = nn.Sequential(
        #        nn.Conv2d(self.in_channels, out_channels * block.expansion, 
        #                  kernel_size=1, stride=stride, bias=False),
        #        nn.BatchNorm2d(out_channels * block.expansion)
        #    )

        layers = []
        layers.append(block(in_channels, out_channels, stride, downsample))
        # self.in_channels = out_channels * block.expansion

        for _ in range(1, blocks):
            layers.append(block(in_channels, out_channels))

        return nn.Sequential(*layers)

    def _initialize_weights(self):

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        if x.dim() == 3:
        # If x is (N, H, W), add a channel dimension
            x = x.unsqueeze(1)
        elif x.dim() == 2:
            # If x is (N, D), reshape to (N, 1, H, W)
            x = x.view(x.size(0), 1, int(math.sqrt(x.size(1))), int(math.sqrt(x.size(1))))
        elif x.dim() == 4:
            # If x is already (N, C, H, W), ensure C=1
            if x.size(1) != 50:
                raise ValueError("Expected input with 1 channel, but got {} channels.".format(x.size(1)))
        else:
            raise ValueError("Unsupported input dimensions: {}".format(x.shape))

        # Initial convolution layers
        x = self.conv1(x)  #makes [64,25,8192]
        x = self.bn1(x)
        x = self.relu(x)

        # Residual layers
        x = self.layer1(x)
        x = self.conv2(x)
        x = self.layer2(x)
        x = self.conv3(x)
        x = self.layer3(x)
        x = self.conv4(x)
        x = self.layer4(x)

        # Global pooling and classification
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
    
def train_and_validate(model, train_loader, val_loader, metadata_path, volume_path,
                       epochs, 
                       lr, 
                       sr,
                       std,
                       device=None):

    # Device configuration
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.BCELoss()  # Binary Cross Entropy for binary classification
    optimizer = torch.optim.Adam(model.parameters(), lr)

    # Training history tracking
    history = {
        'train_loss': [],
        'test_loss': [],
        'test_accuracy': []
    }

    preprocessor=Preprocesing(metadata_path, volume_path,sr)

    # Training loop
    for epoch in range(int(epochs)):
        # Training phase
        model.train()
        total_train_loss = 0.0
        
        for batch_idx, batch_data in enumerate(train_loader):
            # get metadata and indices from the batch
            # metadata = batch_data['metadata']
            # Move data to device
            #inputs = inputs.to(device)
            #targets = targets.float().to(device)
            #preprocesamiento 
            subs_tensor, labels_tensor = preprocessor.process_data(batch_data,std)

            # Move data to device
            inputs = subs_tensor.to(device)
            targets = labels_tensor.to(device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs).squeeze()
            # Normalize targets to [0,1] range instead of adding 1
            outputs = torch.sigmoid(outputs)
           # targets_normalized = targets.float() / targets.max()  
            loss = criterion(outputs, targets)
            #outputs = torch.sigmoid(outputs)
            
            # Compute loss
           # loss = criterion(outputs, targets + 1)
            #loss = torch.mean(torch.abs(outputs - targets))
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        # Compute average training loss
        avg_train_loss = total_train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)

        # Validation phase
        model.eval()
        total_test_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        all_targets = []
        all_predictions = []
    
        with torch.no_grad():
            for batch_data in val_loader:
            
                #metadata = batch_data['metadata']
                subs_tensor, labels_tensor = preprocessor.process_data(batch_data,std)
                
                # Move data to device
                inputs = subs_tensor.to(device)
                targets = labels_tensor.to(device)

                # Forward pass
                outputs = model(inputs).squeeze()
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

        # Compute confusion matrix
        cm = confusion_matrix(all_targets, all_predictions)

        # Convert to percentages
        cm_percentage = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

        # Plot confusion matrix
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm_percentage, annot=True, fmt=".2f", cmap="Blues", xticklabels=["Negative","Positive"], yticklabels=["Negative","Positive"])
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title("Confusion Matrix (Percentages)")

        # Save the figure with epoch number
        conf_mat_filename = f"conf_mat_epoch{epoch+1}.png"
        plt.savefig(conf_mat_filename, bbox_inches='tight', dpi=300)
        plt.close()  # Close the figure to free memory     
      
        # Compute metrics
        avg_test_loss = total_test_loss / len(val_loader)
        test_accuracy = correct_predictions / total_samples * 100

        # Store metrics
        history['test_loss'].append(avg_test_loss)
        history['test_accuracy'].append(test_accuracy)

        # Print epoch summary
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Test Loss: {avg_test_loss:.4f}")
        print(f"Test Accuracy: {test_accuracy:.2f}%\n")

    return history

def main(metadata_path, volume_path, batch_size,train_split,epochs,lr, sr,std,weights_path):
    # Load data
    train_loader, val_loader, test_loader = mrcs_dataloader(metadata_path ,batch_size,train_split)
    
    # Initialize model
    model = ResNet(block=BottleneckBlock, layers=[3, 4, 6, 3], )
    #model=VGG16(1)
    
    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
   #train and validate
    
    history = train_and_validate(
        model, 
        train_loader, 
        val_loader, 
        metadata_path,
        volume_path,
        epochs, 
        lr,
        sr,
        std
    )

    torch.save(model.state_dict(),weights_path)

    np.save('train_loss.npy', np.array(history['train_loss']))
    np.save('test_loss.npy', np.array(history['test_loss']))
    
    
    #Plot the loss curve 
    plt.plot(range(epochs),history['test_loss'], color='r', label='Test loss')
    plt.plot(range(epochs),history['train_loss'], color='b', label='train loss')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss curve")
    plt.legend()
    # Save the figure with dynamic filename
    loss_curve_filename = f"loss_curve.png"
    plt.savefig(loss_curve_filename, bbox_inches='tight', dpi=300)
    plt.close()  # Close the figure
    
    test_model = ResNet(block=BottleneckBlock, layers=[3, 4, 6, 3]) #our model 
    test_model.load_state_dict(torch.load(weights_path)) #load the presaved weights
    test_model = test_model.to(device)

    test_predict(test_model, test_loader, metadata_path, volume_path, sr,std, device=None)


if __name__ == "__main__":
    import argparse
 	# Input parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--md_file', type=str, required=True) #metadata file 
    parser.add_argument('--vol_file', type=str, required=True) #volume file 
    parser.add_argument('--batch_size', type=int, required=True) #batch size is halved for trainigng and normal for test
    parser.add_argument('--split_fraction', type=float, required=True) #splita fraction (max 1)
    parser.add_argument('--epochs', type=int, required=True) #number of epochs
    parser.add_argument('--lr', type=float, required=True) # learning rate
    parser.add_argument('--sr', type=float, required=True) #sampling rate of the ctf 
    parser.add_argument('--std', type=float, required=True) #standard deviation of the wrong alignment
    parser.add_argument('--weights_path', type=str, required=True) #name of the file where you store the weights
    args = parser.parse_args()

    main(args.md_file,args.vol_file, args.batch_size,args.split_fraction, args.epochs, args.lr, args.sr, args.std, args.weights_path)

    
    #python -m pdb 'C:\Users\Livia\Documents\CNB\Missalignment_detector\missalignment_detector.py' --md_file 'C:\Users\Livia\Documents\CNB\Missalignment_detector\input_particles_ctf.xmd' --vol_file 'C:\Users\Livia\Documents\CNB\Missalignment_detector\volume.mrc' --batch_size 100 --split_fraction 0.8 --epochs 10 --lr 1e-4 --sr 1 --weights_path 'C:\Users\Livia\Documents\CNB\Missalignment_detector\ResNet_ribosome.pth'
    


