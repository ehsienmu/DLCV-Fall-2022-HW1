import torch
import os

from torch.utils.data import DataLoader, ConcatDataset
import torch.optim as optim 
import torch.nn as nn

from p1_model import myCNN, myResnet
from p1_datasets import *#get_cifar10_train_val_set, get_cifar10_unlabeled_set
from p1_tool import train, fixed_seed
from torchvision import models

# Modify config if you are conducting different models
# from cfg import LeNet_cfg as cfg
# from cfg import myResnet_cfg as cfg
from p1_model_cfg import mycnn_cfg, pretrained_resnet50_cfg, myresnet_cfg
import argparse

# from prettytable import PrettyTable

# def count_parameters(model):
#     table = PrettyTable(["Modules", "Parameters"])
#     total_params = 0
#     for name, parameter in model.named_parameters():
#         if not parameter.requires_grad: continue
#         params = parameter.numel()
#         table.add_row([name, params])
#         total_params+=params
#     print(table)
#     print(f"Total Trainable Params: {total_params}")
#     return total_params
    

means = [0.485, 0.456, 0.406]
stds = [0.229, 0.224, 0.225]
train_tfm = transforms.Compose([
            ## TO DO ##
            # You can add some transforms here
            AutoAugment(),
            # AutoAugment(AutoAugmentPolicy.CIFAR10),
            transforms.RandomRotation(15),
            transforms.RandomHorizontalFlip(p=0.5),
            # transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0),
            transforms.RandomAffine(0, None, (0.7, 1.3)),
            # ToTensor is needed to convert the type, PIL IMG,  to the typ, float tensor.  
            transforms.ToTensor(),
            
            # experimental normalization for image classification 
            transforms.Normalize(means, stds),
        ])

# train_tfm = transforms.Compose([
#             ## TO DO ##
#             transforms.Resize(232),
#             transforms.CenterCrop(224),
#             transforms.ToTensor(),
            
#             # experimental normalization for image classification 
#             transforms.Normalize(means, stds),
#         ])

val_tfm = transforms.Compose([
            ## TO DO ##
            # transforms.Resize(232),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(means, stds),
        ])

unlabel_tfm = transforms.Compose([
            ## TO DO ##
            transforms.Resize(232),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(means, stds),
        ])

# unlabel_tfm = transforms.RandomChoice( 
#     [
#         transforms.Compose([
#             transforms.RandomHorizontalFlip(),
#             transforms.ColorJitter(0.1, 0.1),
#             transforms.RandomAffine(0, None, (0.8, 1.2)),
#             transforms.ToTensor(),
#         ]),
        
#         transforms.Compose([
#             transforms.RandomHorizontalFlip(),
#             AutoAugment(),
#             transforms.RandomAffine(0, None, (0.8, 1.2)),
#             transforms.ToTensor(),
#         ])
#     ]
# )

def train_interface(cfg):
    
    # parser = argparse.ArgumentParser(description='train_interface of hw2_2 main')
    # parser.add_argument('--model', default='pretrained_resnet50', type=str, help='training model')
    # args = parser.parse_args()

    print(cfg)
    # """ input argumnet """

    training_data_path = cfg['training_data_path']
    val_data_path = cfg['val_data_path']
    model_type = cfg['model_type']
    num_out = cfg['num_out']
    num_epoch = cfg['num_epoch']
    early_stop = cfg['early_stop']
    split_ratio = cfg['split_ratio']
    seed = cfg['seed']
    
    # fixed random seed
    fixed_seed(seed)
    

    os.makedirs(os.path.join('./acc_log',  model_type), exist_ok=True)
    os.makedirs(os.path.join('./save_dir', model_type), exist_ok=True)    
    log_path = os.path.join('./acc_log', model_type, 'acc_' + model_type + '_.log')
    save_path = os.path.join('./save_dir', model_type)



    # with open(log_path, 'w'):
    #     pass
    
    ## training setting ##
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cpu') 
    
    
    """ training hyperparameter """
    # lr = cfg['lr']
    batch_size = cfg['batch_size']
    # milestones = cfg['milestones']
    do_semi = cfg['do_semi']
    
    ## Modify here if you want to change your model ##


    if cfg['model_type'] == 'myCNN':
        model = myCNN(num_out=num_out)
    elif cfg['model_type'] == 'pretrained_resnet50':
        model = models.resnet50(weights='ResNet50_Weights.DEFAULT')
        model.fc = nn.Sequential(nn.Linear(2048, num_out))
    elif cfg['model_type'] == 'myresnet':
        model = myResnet(num_out=num_out)

    # print model's architecture
    print(model)
    print(cfg)
    # Get your training Data 
    ## TO DO ##
    # You need to define your cifar10_dataset yourself to get images and labels for earch data
    # Check myDatasets.py 
      
    train_set = CustomImageDataset(data_folder_path=training_data_path, have_label=True, transform=train_tfm)
    val_set = CustomImageDataset(data_folder_path=val_data_path, have_label=True, transform=val_tfm)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    # if do_semi:
    #     unlabeled_set_1 = get_cifar10_unlabeled_set("./p2_data/unlabeled")  
    #     # unlabeled_loader = DataLoader(unlabeled_set_1, batch_size=batch_size, shuffle=True)
    #     unlabeled_set_2 = get_cifar10_unlabeled_set("./p2_data/public_test")  
    #     unlabeled_set = ConcatDataset([unlabeled_set_1, unlabeled_set_2])
    #     unlabeled_loader = DataLoader(unlabeled_set, batch_size=batch_size, shuffle=True)
    if do_semi:
        unlabeled_set = CustomImageDataset(data_folder_path=training_data_path, have_label=False, transform=train_tfm)
        # unlabeled_loader = DataLoader(unlabeled_set_1, batch_size=batch_size, shuffle=True)
        # unlabeled_set_2 = get_cifar10_unlabeled_set("./p2_data/public_test")  
        # unlabeled_set = ConcatDataset([unlabeled_set_1, unlabeled_set_2])
        unlabeled_loader = DataLoader(unlabeled_set, batch_size=batch_size, shuffle=True)
    else:
        unlabeled_set = None
    
    # define your loss function and optimizer to unpdate the model's parameters.
    optimizer = getattr(torch.optim, cfg['optimizer'])(model.parameters(), **cfg['optim_hparas'])
    ### optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-6, nesterov=True)
    ### optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

    # 沒進步就5個epoch下降 #https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html#torch.optim.lr_scheduler.ReduceLROnPlateau
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, cooldown=3)

    # We often apply crossentropyloss for classification problem. Check it on pytorch if interested
    criterion = nn.CrossEntropyLoss()
    
    # Put model's parameters on your device
    model = model.to(device)
    
    ### TO DO ### 
    # Complete the function train
    # Check tool.py
    # count_parameters(model)
    train(model=model, train_loader=train_loader, val_loader=val_loader, 
          num_epoch=num_epoch, early_stop=early_stop, log_path=log_path, save_path=save_path,
          device=device, criterion=criterion, optimizer=optimizer, scheduler=scheduler, batch_size = batch_size, do_semi = do_semi, unlabeled_set = unlabeled_set, train_set = train_set)

    
if __name__ == '__main__':
    train_interface(cfg = myresnet_cfg)

