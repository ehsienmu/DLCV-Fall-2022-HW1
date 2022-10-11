

import torch
from torch.utils.data.dataset import Dataset
import numpy as np 

from torchvision.transforms import transforms
from torchvision.transforms import AutoAugment
from torchvision.transforms import AutoAugmentPolicy
from PIL import Image
from torchvision.datasets import DatasetFolder

import glob
import os
import argparse
# import scipy.ndimage
import imageio

from p2_tool import read_masks


class SatImageDataset(Dataset):
    def __init__(self, filepath, transform=None):
        self.filenames = []
        self.root = filepath
        self.transform = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])

        if transform is not None:
            self.transform = transform

        files = [f for f in os.listdir(self.root) if f.endswith('.jpg')]
        for i, img_filename in enumerate(files):
            file_prefix = img_filename.split('_')[0]
            mask_filename = os.path.join(self.root, file_prefix + '_mask.png')
            # TODO: remove file_prefix
            self.filenames.append((os.path.join(self.root, img_filename), mask_filename, file_prefix))

        self.len = len(self.filenames)

    def __getitem__(self, idx):

        img_filename, mask_filename, file_prefix = self.filenames[idx]
        img = Image.open(img_filename).convert("RGB")
        img = self.transform(img)

        mask = read_mask(mask_filename)

        return img, torch.tensor(mask).long(), file_prefix

    def __len__(self):
        return self.len


class SatImageTestDataset(Dataset):
    def __init__(self, filepath, transform=None):
        self.filenames = []
        self.root = filepath
        self.transform = transforms.Compose(
            [transforms.ToTensor()]#, transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
        )

        if transform is not None:
            self.transform = transform

        files = [f for f in os.listdir(self.root) if f.endswith('.jpg')]
        for i, img_filename in enumerate(files):
            self.filenames.append((os.path.join(self.root, img_filename), img_filename))

        self.len = len(self.filenames)

    def __getitem__(self, idx):

        img_filename, origin_filename = self.filenames[idx]
        img = Image.open(img_filename).convert("RGB")
        img = self.transform(img)

        return img, origin_filename

    def __len__(self):
        return self.len



class CustomImageDataset(Dataset):
    def __init__(self, data_folder_path, have_label, transform=None):
        # path = r'./hw1_data/hw1_data/p1_data/train_50/*.png'
        if(data_folder_path[-1] != '/'):
            data_folder_path += '/'
        mask_filename = glob.glob(data_folder_path+'*.png')
        mask_filename.sort()
        sat_filename = glob.glob(data_folder_path+'*.jpg')
        sat_filename.sort()
        
        # print('mask_filename[:5]:', mask_filename[:5])
        # print('sat_filename[:5]:', sat_filename[:5])
        if have_label:
            self.masks = mask_filename 
        else:
            self.masks = None
            
        self.sats = sat_filename 
        
        # The transform for the image
        self.transform = transform
        
        # prefix of the files' names
        self.prefix = data_folder_path
        print('from', self.prefix)
        print(f'Number of images is {len(self.sats)}')
    
    def __len__(self):
        return len(self.sats)
    
    def __getitem__(self, idx):
        ## TO DO ##
        # You should read the image according to the file path and apply transform to the images
        # Use "PIL.Image.open" to read image and apply transform
        
        # You shall return image, label with type "long tensor" if it's training set
        # pass
        # full_path = os.path.join(self.prefix, self.images[idx])
        

        
        sat = Image.open(self.sats[idx]).convert("RGB")
        seg = imageio.imread(self.masks[idx])
        # print(sat.size[0])
        mask = torch.tensor(read_masks(seg, sat.size))
        # print('type(mask):', type(mask)) # <class 'torch.Tensor'>   
        if self.transform != None:   
            sat = self.transform(sat)
            
        # print('type(sat):', type(sat)) #<class 'PIL.Image.Image'> 

        if self.masks != None:
            #  print(type((transform_img, self.labels[idx])))
            return (sat, mask.long(), self.sats[idx], self.masks[idx])
        else:
            return (sat)