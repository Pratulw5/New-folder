# Setup and Imports
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import nibabel as nib  # For MRI NIfTI files
import umap


# Preprocess and Load MRI Data
class MRIDataset(Dataset): #custom pytorch dataset class for MRI data
    def __init__(self, root_dir, transform=None): #init method to initialize dataset
        self.root_dir = root_dir
        self.transform = transform
        self.files = [f for f in os.listdir(root_dir) if f.endswith(('.nii', '.nii.gz'))]

    def __len__(self): #return the length of the dataset
        return len(self.files)

    def __getitem__(self, idx):# get item method to get a single MRI file
        #idx is the index of the MRI file to retrieve
        file_path = os.path.join(self.root_dir, self.files[idx])
        #load the MRI file using nibabel
        img = nib.load(file_path).get_fdata()
        
        # Take a central slice from the 3D volume along the 3d z-axis
        img = img[:, :, img.shape[2] // 2]
        
        # Converts the slice to float32 and normalizes pixel values to [0, 1].
        img = img.astype(np.float32)
    
        img = (img - img.min()) / (img.max() - img.min())  # Normalize 0-1
        
        #Applies any provided transform (e.g., resizing, tensor conversion).
        # None in this case, but can be set to transforms.Compose([...])
        if self.transform:
            img = self.transform(img)

        return img.unsqueeze(0)  # add channel dimension
    #In image processing with PyTorch, the channel dimension refers to the part of a tensor that represents color channels (like Red, Green, Blue) or, for grayscale images, a single channel.

# Directory path
dataset_dir = "/home/groups/comp3710/OASIS_Preprocessed/"

#Composed of:
# ToTensor(): Converts the image to a PyTorch tensor.
# Resize((64, 64)): Resizes the image to 64x64 pixels.
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((64, 64)),  
])

# An instance of MRIDataset with the specified directory and transform.
dataset = MRIDataset(dataset_dir, transform=transform)

# Wraps the dataset for batch loading (batch size 16, shuffling enabled).
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
#The dataloader provides batches of preprocessed tensors (your MRI slices) to your model during training or inference. You loop over the dataloader in your training code to feed data into your neural network.



# This class builds a VAE that compresses images to a low-dimensional latent space and reconstructs them, using convolutional layers for both encoding and decoding.
class VAE(nn.Module):
    def __init__(self, latent_dim=2):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1),  # 64x64 -> 32x32
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),  # 32x32 -> 16x16
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),  # 16x16 -> 8x8
            nn.ReLU(),
            nn.Flatten()
        )
        self.fc_mu = nn.Linear(128*8*8, latent_dim)
        self.fc_logvar = nn.Linear(128*8*8, latent_dim)
        
        # Decoder
        self.decoder_input = nn.Linear(latent_dim, 128*8*8)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),  # 8x8 -> 16x16
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),  # 16x16 -> 32x32
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, 2, 1),  # 32x32 -> 64x64
            nn.Sigmoid()
        )
        
    def encode(self, x):
        x_enc = self.encoder(x)
        mu = self.fc_mu(x_enc)
        logvar = self.fc_logvar(x_enc)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        x = self.decoder_input(z)
        x = x.view(-1, 128, 8, 8)
        x = self.decoder(x)
        return x
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar