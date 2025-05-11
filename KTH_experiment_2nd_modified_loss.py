import sys
import os
import pickle

# Add the parent directory to sys.path
sys.path.append("/home/jovyan/video-storage/amit_files/image_background_remove_tool")

import torch
device = "cuda" if torch.cuda.is_available else "cpu"

from carvekit.ml.arch.tracerb7.efficientnet import EfficientEncoderB7
from carvekit.ml.wrap.tracer_b7 import TracerUniversalB7

model = TracerUniversalB7()

model = model.to(device)


# Add the parent directory to sys.path
sys.path.append("/home/jovyan/video-storage/amit_files")

# !pip install gym

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
        img3 = model([img2_path])[0]
        
        # Apply any transformations
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)
            
        # Convert grayscale to 3-channel by stacking
        if img1.shape[0] == 1:  # If single channel
            img1 = torch.cat([img1, img1, img1], dim=0)  # Stack channel 3 times
        
        if img2.shape[0] == 1:  # If single channel
            img2 = torch.cat([img2, img2, img2], dim=0)  # Stack channel 3 times

        # if img3.shape[0] == 1:  # If single channel
        #     img3 = torch.cat([img3, img3, img3], dim=0)  # Stack channel 3 times
        
        return img1, img2, img3

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

for i , img in enumerate(train_loader):
    # print(i)
    # print(img[0].shape)
    # print(img[1].shape)
    # print(type(img[1]))
    imge = np.asarray(img[2][0]).transpose((1,2,0))
    plt.imshow(imge)
    break


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


def l2_reconstruction_loss(x, x_, foreground_mask=None, epsilon=1e-6):
    device = x.device
    x_ = x_.to(device)
    pixel_loss = (x - x_) ** 2  # [B, C, H, W]

    if foreground_mask is not None:
        if isinstance(foreground_mask, list):
            if all(isinstance(mask, Image.Image) for mask in foreground_mask):
                foreground_mask = [
                    torch.tensor(np.array(mask), dtype=torch.float32) for mask in foreground_mask
                ]
                foreground_mask = torch.stack(foreground_mask, dim=0)
            else:
                raise ValueError("All elements in foreground_mask list must be PIL Images.")

        foreground_mask = foreground_mask.to(device)

        if len(foreground_mask.shape) == 3:
            foreground_mask = foreground_mask.unsqueeze(1)  # [B, 1, H, W]

        # Broadcast mask to all channels
        foreground_mask = foreground_mask.repeat(1, pixel_loss.shape[1], 1, 1)  # [B, C, H, W]

        if foreground_mask.shape != pixel_loss.shape:
            raise ValueError(f"Foreground mask shape {foreground_mask.shape} does not match loss shape {pixel_loss.shape}")

        # Calculate weighting factor - higher weight for foreground regions
        weight = 1.0 / torch.clamp(1.0 - foreground_mask, min=epsilon)
        
        # Apply weighting to loss
        weighted_loss = pixel_loss * weight
        
        # To maintain reasonable scale, divide by mean weight
        return weighted_loss.mean() / weight.mean()

    return pixel_loss.mean()



criterion = l2_reconstruction_loss

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
            x, x_ ,foreground_mask = data[0].to(args.device), data[1].to(args.device), data[2].to(args.device)
            
            optim.zero_grad()

            # Mixed precision forward pass
            with autocast():
                x_t, z, k, m, p, heatmap = kp_network(x, x_)
                loss = criterion(x_t, x_,foreground_mask = foreground_mask)
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
            x, x_ ,foreground_mask = data[0].to(args.device), data[1].to(args.device), data[2].to(args.device)

            with autocast():
                x_t, z, k, m, p, heatmap = kp_network(x, x_)
                loss = criterion(x_t, x_,foreground_mask = foreground_mask)
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

