import torch
import os
# import random

from torch.utils.data import DataLoader
import torch.optim as optim 
import torch.nn as nn
from PIL import Image
from torchvision.datasets import DatasetFolder

from torch.utils.data.dataset import Dataset
from torchvision import models

from p1_model_cfg import mycnn_cfg, pretrained_resnet50_cfg
import argparse

from torchvision.transforms import transforms
# from p1_datasets import *
import numpy as np
from tqdm import tqdm
import argparse
import glob
import pickle
from p1_model import myCNN


means = [0.485, 0.456, 0.406]
stds = [0.229, 0.224, 0.225]

val_tfm = transforms.Compose([
            ## TO DO ##
            transforms.Resize(232),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(means, stds),
        ])

def load_parameters(model, path):
    print(f'Loading model parameters from {path}...')
    param = torch.load(path)#, map_location={'cuda:0': 'cuda:1'})
    model.load_state_dict(param)
    print("End of loading !!!")

def fixed_seed(myseed):
    np.random.seed(myseed)
    # random.seed(myseed)
    torch.manual_seed(myseed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(myseed)
        torch.cuda.manual_seed(myseed)
        


class CustomImageDataset(Dataset):
    def __init__(self, data_folder_path, have_label, transform=None):
        # path = r'./hw1_data/hw1_data/p1_data/train_50/*.png'
        if(data_folder_path[-1] != '/'):
            data_folder_path += '/'
        images_filename = glob.glob(data_folder_path+'*.png')
        images_filename.sort()
        
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
        # print('self.images[idx]:', self.images[idx])
        # You shall return image, label with type "long tensor" if it's training set
        # pass
        # full_path = os.path.join(self.prefix, self.images[idx])
        img = Image.open(self.images[idx]).convert("RGB")
        transform_img = self.transform(img)
        if self.labels != None:
            #  print(type((transform_img, self.labels[idx])))
            return (transform_img, self.labels[idx], self.images[idx])
        else:
            return (transform_img, self.images[idx])


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--input_dir', default='', type=str)
    parser.add_argument('--output_file', default='', type=str)
    parser.add_argument('--model_file', default='', type=str)

    
    args = parser.parse_args()

    
    # fixed random seed
    fixed_seed(1)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    
    """ training hyperparameter """

    model = myCNN(num_out=50)
    
    load_parameters(model, args.model_file)
    # Put model's parameters on your device
    model = model.to(device)
    
    # train_set = CustomImageDataset(data_folder_path=training_data_path, have_label=True, transform=train_tfm)
    val_set = CustomImageDataset(data_folder_path=args.input_dir, have_label=True, transform=val_tfm)
    # train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    # print(val_set)

    # sdfdf
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False)

    # We often apply crossentropyloss for classification problem. Check it on pytorch if interested
    # criterion = nn.CrossEntropyLoss()
    
    ### TO DO ### 
    # Complete the function train
    # Check tool.py
    # count_parameters(model)
    results = []
    labels = []
    with torch.no_grad():
        model.eval()
        val_loss = 0.0
        corr_num = 0
        val_acc = 0.0
        
        ## TO DO ## 
        # Finish forward part in validation. You can refer to the training part 
        # Note : You don't have to update parameters this part. Just Calculate/record the accuracy and loss. 

        for batch_idx, (data, label, fname,) in enumerate(tqdm(val_loader)):
            # put the data and label on the device
            # note size of data (B,C,H,W) --> B is the batch size
            data = data.to(device)
            label = label.to(device)

            # pass forward function define in the model and get output 
            output_sec_last = model.sec_last_out(data)
            # sdfsdff
            results.append(output_sec_last.cpu().detach().numpy())
            labels.append(label.cpu().detach().numpy())

    # with open('dct_var_record.pk', 'rb') as f:
    #     dct_var = pickle.load(f)
    
    with open('./sec_last_for_tsne_pred_87.pk', 'wb') as f:
        pickle.dump(results, f)
    with open('./sec_last_for_tsne_label_87.pk', 'wb') as f:
        pickle.dump(labels, f)
    print(len(results))
    print(len(labels))
    print(results[0])
    print(labels[:10])

        # scheduler += 1 for adjusting learning rate later
        
        # averaging training_loss and calculate accuracy
        # val_loss = val_loss / len(val_loader.dataset) 
        # val_acc = corr_num / len(val_loader.dataset)
        # print('val acc =', val_acc)
        # record the training loss/acc
        # overall_val_loss[i], overall_val_acc[i] = val_loss, val_acc
    #     overall_val_loss.append(val_loss)
    #     overall_val_acc.append(val_acc)
    #     scheduler.step(val_loss)
    #     # scheduler.step()
    # #####################
        
        # Display the results
        
        # # print(f'training loss : {train_loss:.4f} ', f' train acc = {train_acc:.4f}' )
        # print(f'val loss : {val_loss:.4f} ', f' val acc = {val_acc:.4f}' )
        # print('========================\n')

    # with open(args.output_file, 'w') as f:
    #     f.write('filename,label\n')
    #     for fname, predl in results:
    #         f.write(fname)
    #         f.write(',')
    #         f.write(predl)
    #         f.write('\n')
        