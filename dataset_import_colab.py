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

# Load the data using the ImageFolder dataset class
batch_size=32
min_height=324
min_width=558

train_hr = os.path.join(data_dir,'training_hr')
train_lr = os.path.join(data_dir,'training_lr/X2')
val_hr = os.path.join(data_dir,'validation_hr')
val_lr = os.path.join(data_dir,'validation_lr/X2')
test_hr = os.path.join(data_dir,'test_hr')
test_lr = os.path.join(data_dir,'test_lr/X2')

#parameter: 
# input_folder:the path of folder
# output_folder: the path of folder
# transform: example:transform['train']
#return:input_image_result, output_image_result:torch tensor,shape (height, width, num_channels)
class Div2K(Dataset):
    #input_folder:the path of folder,low resolution
    #output_folder: the path of folder, high resolution
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
#example:size of input_images is (x,y). size of output images is (2*x,2*y)
#CenterCrop for input images: (x-w,y-h)
#CenterCrop for output images: (2*x-2*w,2*y-2*h)
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
dataset = Div2K(train_lr, train_hr, transform_output= transform['train_output'],transform_input=transform['train_input'])
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
