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

triplet_first_image = "/home/jovyan/video-storage/amit_files/MTP_01/Vimeo_Triplet_Folder/second_image"
img_dir = os.listdir(triplet_first_image)

image_dir = [os.path.join(triplet_first_image, image) for image in img_dir]

import time
len(image_dir)

start = time.time()
output = model(image_dir)
print(time.time() - start)

# Save the list to a pickle file
with open('model_outputs.pkl', 'wb') as file:
    pickle.dump(output, file)

