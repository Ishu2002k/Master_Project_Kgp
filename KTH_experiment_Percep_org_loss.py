import sys
import os
import pickle

# Add the parent directory to sys.path
sys.path.append("/home/jovyan/video-storage/amit_files")

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.optim import Adam
from torch.cuda.amp import autocast, GradScaler

from keypoints.keypoints.data_augments import TpsAndRotate, nop
from keypoints.keypoints.models import keynet
from keypoints.keypoints.utils import ResultsLogger
from keypoints.keypoints.ds import datasets as ds
from keypoints.keypoints.config import config

from keypoints.keypoints.ds.datasets import split

import yaml
from PIL import Image
import argparse
from pathlib import Path
import torch
import json
import numpy as np
from tqdm import tqdm

args_path = '/home/jovyan/video-storage/amit_files/Master_Thesis_Project/keypoints_KTH.yml'

torch.cuda.set_device(0)

args = config(['--config', args_path,'--data_root','/home/jovyan/video-storage/amit_files/Master_Thesis_Project/KTH_Dataset_2/paired_images'])
print(args)

run_dir = f'data/models/keypoints/{args.model_type}/run_{args.run_id}'
run_dir

display = ResultsLogger(run_dir=run_dir,
                            num_keypoints=args.model_keypoints,
                            title = 'Results',
                            visuals = args.display,
                            image_capture_freq = args.display_freq,
                            kp_rows = args.display_kp_rows,
                            comment = args.comment)
display.header(args)


import os
import random
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import torch

class KthDataset(Dataset):
    def __init__(self, dir1, dir2, model=None, transform=None, shuffle=False):
        self.dir1 = os.path.join(args.data_root, args.dataset_1)
        self.dir2 = os.path.join(args.data_root, args.dataset_2)
        self.transform = transform
        self.model = model
        
        # Get image lists
        self.images1 = sorted(os.listdir(self.dir1))
        self.images2 = sorted(os.listdir(self.dir2))
        
        assert len(self.images1) == len(self.images2), "Both directories must contain the same number of images."
        
        # Create pairs of image paths
        self.image_pairs = list(zip(self.images1, self.images2))
        
        # Shuffle the pairs if requested
        if shuffle:
            random.shuffle(self.image_pairs)
            self.images1, self.images2 = zip(*self.image_pairs)

    def __len__(self):
        return len(self.images1)

    def __getitem__(self, idx):
        img1_path = os.path.join(self.dir1, self.images1[idx])
        img2_path = os.path.join(self.dir2, self.images2[idx])
        
        # Open images
        img1 = Image.open(img1_path)
        img2 = Image.open(img2_path)
        
        # Apply any transformations
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            
        # Convert grayscale to 3-channel by stacking
        if img1.shape[0] == 1:  # If single channel
            img1 = torch.cat([img1, img1, img1], dim=0)  # Stack channel 3 times
        
        if img2.shape[0] == 1:  # If single channel
            img2 = torch.cat([img2, img2, img2], dim=0)  # Stack channel 3 times
        
        return img1, img2

def split(dataset, train_len, test_len):
    """
    Split a dataset into training and testing subsets.
    """
    assert train_len + test_len <= len(dataset), "Sum of train and test lengths exceeds dataset size"
    
    train_dataset, test_dataset = random_split(
        dataset, 
        [train_len, test_len],
        generator=torch.Generator().manual_seed(42)
    )
    
    return train_dataset, test_dataset


pin_memory = False if args.device == 'cpu' else True

# Define your transformation
Kth_transform = transforms.Compose([
    transforms.Lambda(lambda img: img.resize((img.size[0], img.size[1]), resample=Image.LANCZOS)),
    transforms.ToTensor(),
])

# Create the dataset with shuffling
dataset = KthDataset(
    args.dataset_1, 
    args.dataset_2, 
    model=None, 
    transform = Kth_transform,
    shuffle=True  # Enable shuffling at dataset creation
)

# Split into train and test
train_dataset, test_dataset = split(dataset, args.dataset_train_len, args.dataset_test_len)


# Create DataLoaders with additional shuffling
train_loader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True,  # Shuffle again during training
    drop_last=True,
    pin_memory=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=args.batch_size,
    shuffle=False,  # No need to shuffle test data
    drop_last=True,
    pin_memory=True
)


""" model """
kp_network = keynet.make(args).to(args.device)

print(kp_network)

optim = Adam(kp_network.parameters(), lr=1e-4)

""" data augmentation """
if args.data_aug_type == 'tps_and_rotate':
    augment = TpsAndRotate(args.data_aug_tps_cntl_pts, args.data_aug_tps_variance, args.data_aug_max_rotate)
else:
    augment = nop

import matplotlib.pyplot as plt
scaler = GradScaler()
threshold = 0.5

# # loss function
# def l2_reconstruction_loss(x, x_, loss_mask=None):
#     loss = (x - x_) ** 2
#     if loss_mask is not None:
#         loss = loss * loss_mask
#     return torch.mean(loss)

# criterion = l2_reconstruction_loss

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.transforms import Normalize

class VGGPerceptualLoss(nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        # Load pretrained VGG19 and freeze parameters
        vgg = models.vgg19(pretrained=True).features
        self.vgg = nn.Sequential()
        for i in range(36):  # Use features up to relu5_4
            self.vgg.add_module(str(i), vgg[i])
        for param in self.vgg.parameters():
            param.requires_grad = False
        self.vgg.eval()
        
        # For normalizing input images
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        
        self.resize = resize
        self.criterion = nn.MSELoss()
        
    def _normalize(self, x):
        # Normalize to match VGG input distribution
        return (x - self.mean) / self.std
    
    def forward(self, x, x_, loss_mask=None):
        """
        x, x_: Input images (original and reconstructed)
        loss_mask: Optional mask to apply to the loss
        """
        # Handle grayscale input by repeating channels
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
            x_ = x_.repeat(1, 3, 1, 1)
            
        # Ensure proper scale [0,1]
        if x.min() < 0:
            x = (x + 1) / 2
            x_ = (x_ + 1) / 2
            
        # Normalize inputs
        x = self._normalize(x)
        x_ = self._normalize(x_)
        
        # Extract features
        features_x = self.vgg(x)
        features_x_ = self.vgg(x_)
        
        # Calculate loss
        loss = self.criterion(features_x, features_x_)
        
        # Apply mask if provided
        if loss_mask is not None:
            # Need to resize mask to match feature dimensions
            if self.resize:
                mask = nn.functional.interpolate(loss_mask, size=features_x.shape[2:], mode='nearest')
            else:
                mask = loss_mask
            loss = loss * mask
            
        return loss.mean()

# # Usage
# perceptual_loss = VGGPerceptualLoss().to(device)  # Move to same device as your tensors
# loss = perceptual_loss(original_image, reconstructed_image, loss_mask)

criterion = VGGPerceptualLoss().to(args.device)

def to_device(data, device):
    return tuple([x.to(device) for x in data])


# Initialize lists to store losses for each epoch
train_losses = []
val_losses = []

# Initialize the loss file before training
loss_file_path = run_dir + '/losses.json'
with open(loss_file_path, 'w') as f:
    json.dump({'train_losses': [], 'val_losses': []}, f)

# Main training loop
for epoch in tqdm(range(0, args.epochs)):
    epoch_train_loss = 0
    epoch_val_loss = 0
    num_train_batches = 0
    num_val_batches = 0

    if not args.demo:
        # Training
        batch = tqdm(train_loader, total=len(train_dataset) // args.batch_size)
        for i, data in enumerate(batch):
            data = to_device(data, device=args.device)
            x, x_ = data[0].to(args.device), data[1].to(args.device)
            
            optim.zero_grad()

            # Mixed precision forward pass
            with autocast():
                x_t, z, k, m, p, heatmap = kp_network(x, x_)
                loss = criterion(x_t, x_,loss_mask = None)
                epoch_train_loss += loss.item()
                num_train_batches += 1

            # Scaled backward pass
            scaler.scale(loss).backward()
            scaler.step(optim)    # Update model parameters
            scaler.update()       # Adjust the scale for next iteration

            if i % args.checkpoint_freq == 0:
                kp_network.save(run_dir + '/checkpoint')
                
            display.log(batch, epoch, i, loss, optim, x, x_, x_t, heatmap, k, m, p, loss_mask=None, type='train', depth=20)

    # Calculate average train loss for the epoch
    epoch_train_loss /= max(1, num_train_batches)
    train_losses.append(epoch_train_loss)

    # Validation
    with torch.no_grad():
        batch = tqdm(test_loader, total=len(test_dataset) // args.batch_size)
        for i, data in enumerate(batch):
            data = to_device(data, device=args.device)
            x, x_ = data[0].to(args.device), data[1].to(args.device)

            with autocast():
                x_t, z, k, m, p, heatmap = kp_network(x, x_)
                loss = criterion(x_t, x_,loss_mask = None)
                epoch_val_loss += loss.item()
                num_val_batches += 1

            display.log(batch, epoch, i, loss, optim, x, x_, x_t, heatmap, k, m, p, loss_mask=None, type='test', depth=20)

    # Calculate average validation loss for the epoch
    epoch_val_loss /= max(1, num_val_batches)
    val_losses.append(epoch_val_loss)

    # Save the updated train and val losses after each epoch
    with open(loss_file_path, 'r+') as f:
        loss_data = json.load(f)
        loss_data['train_losses'].append(epoch_train_loss)
        loss_data['val_losses'].append(epoch_val_loss)
        f.seek(0)  # Go to the beginning of the file
        json.dump(loss_data, f, indent=4)  # Write updated data
        f.truncate()  # Remove any leftover data from previous writes

    ave_loss, best_loss = display.end_epoch(epoch, optim)

    # Save if model improved
    if ave_loss <= best_loss and not args.demo:
        kp_network.save(run_dir + '/best')
