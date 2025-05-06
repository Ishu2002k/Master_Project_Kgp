"""Vimeo_Triplet.ipynb"""

import sys

# Add the parent directory to sys.path
sys.path.append("/home/jovyan/video-storage/amit_files/")


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
import argparse
from pathlib import Path
import torch
import json
import numpy as np
from tqdm import tqdm

args_path = '/home/jovyan/video-storage/amit_files/MTP_01/keypoints_triplet.yaml'

torch.cuda.set_device(0)

args = config(['--config', args_path,'--data_root','/home/jovyan/video-storage/amit_files/MTP_01/Vimeo_Triplet_Folder/'])
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
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt

class TripletDataset(Dataset):
    def __init__(self, dir1, dir2, transform=None):
        self.dir1 = os.path.join(args.data_root,args.dataset_1)
        self.dir2 = os.path.join(args.data_root,args.dataset_2)
        self.transform = transform

        # Ensure directories contain the same number of images
        self.images1 = sorted(os.listdir(self.dir1))
        self.images2 = sorted(os.listdir(self.dir2))

        assert len(self.images1) == len(self.images2), "Both directories must contain the same number of images."

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

        return img1, img2

triplet_transform = transforms.Compose([
    transforms.Lambda(lambda img: img.resize((img.size[0]//2,img.size[1]//2), resample=Image.LANCZOS)),  # Lanczos resampling
    transforms.ToTensor(),
])

pin_memory = False if args.device == 'cpu' else True

Dataset = TripletDataset(args.dataset_1,args.dataset_2,triplet_transform)

train_t,test_t = split(Dataset,args.dataset_train_len, args.dataset_test_len)

trip_train = DataLoader(train_t, batch_size = args.batch_size, shuffle=True, drop_last=True, pin_memory=pin_memory)
trip_test = DataLoader(test_t, batch_size = args.batch_size, shuffle=True, drop_last=True, pin_memory=pin_memory)

""" model """
kp_network = keynet.make(args).to(args.device)

kp_network

optim = Adam(kp_network.parameters(), lr=1e-4)

scaler = GradScaler()

# loss function
def l2_reconstruction_loss(x, x_, loss_mask=None):
    loss = (x - x_) ** 2
    if loss_mask is not None:
        loss = loss * loss_mask
    return torch.mean(loss)

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
for epoch in tqdm(range(1, args.epochs + 1)):
    epoch_train_loss = 0
    epoch_val_loss = 0
    num_train_batches = 0
    num_val_batches = 0

    if not args.demo:
        # Training
        batch = tqdm(trip_train, total=len(train_t) // args.batch_size)
        for i, data in enumerate(batch):
            data = to_device(data, device=args.device)
            x, x_ = data[0].to(args.device), data[1].to(args.device)

            optim.zero_grad()

            # Mixed precision forward pass
            with autocast():
                x_t, z, k, m, p, heatmap = kp_network(x, x_)
                loss = criterion(x_t, x_, loss_mask=None)
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
        batch = tqdm(trip_test, total=len(test_t) // args.batch_size)
        for i, data in enumerate(batch):
            data = to_device(data, device=args.device)
            x, x_ = data[0].to(args.device), data[1].to(args.device)

            with autocast():
                x_t, z, k, m, p, heatmap = kp_network(x, x_)
                loss = criterion(x_t, x_, loss_mask=None)
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











