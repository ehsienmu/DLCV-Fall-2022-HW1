

import torch
from torch.utils.data.dataset import Dataset
import numpy as np 

from torchvision.transforms import transforms
from torchvision.transforms import AutoAugment
from torchvision.transforms import AutoAugmentPolicy
from PIL import Image
from torchvision.datasets import DatasetFolder

import glob

means = [0.485, 0.456, 0.406]
stds = [0.229, 0.224, 0.225]
unlabel_tfm = transforms.Compose([
            ## TO DO ##
            transforms.Resize(232),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(means, stds),
        ])


class PsudeoDataset(Dataset):
    def __init__(self, dataset, labels, tfm=unlabel_tfm):
        self.data = dataset
        self.labels = labels
        self.tfm = tfm

    def __getitem__(self, idx):
        img = self.data[idx][0]
        img = transforms.ToPILImage()(img).convert("RGB")
        return self.tfm(img), self.labels[idx]

    def __len__(self):
        return len(self.labels)

# def get_cifar10_train_val_set(root, prefix, ratio=0.9, cv=0):
    
#     # get all the images path and the corresponding labels
#     with open(root, 'r') as f:
#         data = json.load(f)
#     images, labels = data['images'], data['categories']

    
#     info = np.stack( (np.array(images), np.array(labels)) ,axis=1)
#     N = info.shape[0]

#     # apply shuffle to generate random results 
#     np.random.shuffle(info)
#     x = int(N*ratio) 
    
#     all_images, all_labels = info[:,0].tolist(), info[:,1].astype(np.int32).tolist()


#     train_image = all_images[:x]
#     val_image = all_images[x:]

#     train_label = all_labels[:x] 
#     val_label = all_labels[x:]
    

    
#     ## TO DO ## 
#     # Define your own transform here 
#     # It can strongly help you to perform data augmentation and gain performance
#     # ref: https://pytorch.org/vision/stable/transforms.html
#     train_transform = train_tfm
  
#     # normally, we dont apply transform to test_set or val_set
#     val_transform = val_tfm

 
  
#     ## TO DO ##
#     # Complete class cifiar10_dataset
#     train_set, val_set = cifar10_dataset(images=train_image, labels=train_label, transform=train_transform, prefix = prefix), \
#                         cifar10_dataset(images=val_image, labels=val_label, transform=val_transform, prefix = prefix)


#     return train_set, val_set




# def get_cifar10_unlabeled_set(root, ratio=0.9, cv=0):
#     # return class cifar10_unlabeled_dataset(Dataset):
#     # return DatasetFolder(root, loader=lambda x: Image.open(x), extensions="jpg", transform=unlabel_tfm)
#     images = []
        
#     files = os.listdir(root)

#     for file in files:
#         # make sure file is an image
#         if file.endswith(('.jpg', '.png', 'jpeg')):
#             # img_path = self.prefix + file
#             images.append(file)
        
        
#     unlabeled_set = cifar10_dataset(images=images, prefix = root, transform=unlabel_tfm)


#     return unlabeled_set


## TO DO ##
# Define your own cifar_10 dataset
class CustomImageDataset(Dataset):
    def __init__(self, data_folder_path, have_label, transform=None):
        # path = r'./hw1_data/hw1_data/p1_data/train_50/*.png'
        if(data_folder_path[-1] != '/'):
            data_folder_path += '/'
        images_filename = glob.glob(data_folder_path+'*.png')

        # print(images_filename)
        if have_label:
            labels = []
            for full_path in images_filename:
                labels.append(int(full_path.split('/')[-1].split('_')[0]))
            # print('labels[:5]:', labels[:5])
            self.labels = torch.tensor(labels)
        else:
            self.labels = None

        # It loads all the images' file name and correspoding labels here
        self.images = images_filename 
        
        # The transform for the image
        self.transform = transform
        
        # prefix of the files' names
        self.prefix = data_folder_path
        print('from', self.prefix)
        print(f'Number of images is {len(self.images)}')
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        ## TO DO ##
        # You should read the image according to the file path and apply transform to the images
        # Use "PIL.Image.open" to read image and apply transform
        
        # You shall return image, label with type "long tensor" if it's training set
        # pass
        # full_path = os.path.join(self.prefix, self.images[idx])
        img = Image.open(self.images[idx]).convert("RGB")
        transform_img = self.transform(img)
        if self.labels != None:
            #  print(type((transform_img, self.labels[idx])))
            return (transform_img, self.labels[idx])
        else:
            return (transform_img)