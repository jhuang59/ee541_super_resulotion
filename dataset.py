
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import os, shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils as utils
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from PIL import Image
from PIL import ImageOps
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
#parameter: 
# input_folder:the path of folder
# output_folder: the path of folder
# transform: example:transform['train']
#return:input_image_result, output_image_result:torch tensor,shape (height, width, num_channels)
class Div2K(Dataset):
    #input_folder:the path of folder
    #output_folder: the path of folder
    #transform:transform
    #input_images_path: a list of paths of single images
    #output_images_path: a list of paths of single images
  def __init__(self, input_folder, output_folder, transform_output,transform_input,):
    self.input_folder = input_folder
    self.output_folder = output_folder
    self.transform_in = transform_input
    self.transform_out = transform_output
    self.input_images_path=sorted([os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith('.png')])
    self.output_images_path=sorted([os.path.join(output_folder, f) for f in os.listdir(output_folder) if f.endswith('.png')])
    
  def __len__(self):
    return len(self.input_images_path)
    #input_image:image object, size(1356, 2040)
    #output_image:image object, size(1356, 2040)
    #input_image_transformed:image object, size (558, 324)
    #output_image_transformed:image object, size (558, 324)
    #input_image_result: torch.Size([324, 558, 3]), example:(height,width,num_channels)
  def __getitem__(self,index):
    input_image = Image.open(self.input_images_path[index]).convert('RGB')
    output_image = Image.open(self.output_images_path[index]).convert('RGB')
    input_image_transformed = self.transform_in(input_image)
    output_image_transformed = self.transform_out(output_image)
    input_image_result = torch.tensor(np.array(input_image_transformed), dtype=torch.float32) / 255.
    output_image_result = torch.tensor(np.array(output_image_transformed), dtype=torch.float32) / 255.
    
    return input_image_result, output_image_result


class SuperResolutionDataset(Dataset):
    def __init__(self, image_paths, lr_scale):
        self.image_paths = image_paths
        self.lr_scale = lr_scale

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        hr_image = Image.open(self.image_paths[idx]).convert('RGB')
        w, h = hr_image.size
        lr_image = hr_image.resize((w // self.lr_scale, h // self.lr_scale), resample=Image.BICUBIC)
        hr_image = torch.tensor(np.array(hr_image), dtype=torch.float32) / 255.
        lr_image = torch.tensor(np.array(lr_image), dtype=torch.float32) / 255.
        
        return lr_image, hr_image

min_height=324
min_width=558

transform = {
    'train_input': transforms.Compose([
        transforms.CenterCrop([min_height//2,min_width//2]),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        ]),
    'train_output': transforms.Compose([
        transforms.CenterCrop([min_height,min_width]),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        ]),
    'val_test': transforms.Compose([
        transforms.CenterCrop([min_height,min_width]),
        transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
    ])
}
# Example usage:

folder_path = "/Users/huangjie/Documents/hw7ee_541/training_lr/X2"
folder_path2 = "/Users/huangjie/Documents/hw7ee_541/training_hr"

image_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.JPG')]
scale_factor = 4
batch_size = 32

dataset = Div2K(folder_path, folder_path2, transform['train_output'],transform_input=transform['train_input'])
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

i=0
for lr_images,hr_images in dataloader:
    # Plot the first image in the batch
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    #expected shape (height, width, num_channels)
    ax[0].imshow(lr_images[0])
    ax[0].set_title('Low-resolution')
    ax[1].imshow(hr_images[0])
    ax[1].set_title('High-resolution')
    plt.show()
    i+=1
    if(i>2):
        break