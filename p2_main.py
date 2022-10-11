import torch
import os

from torch.utils.data import DataLoader, ConcatDataset
import torch.optim as optim 
import torch.nn as nn
from tqdm import tqdm

from p2_model import *
from p2_datasets import *
from p2_tool import train, fixed_seed
from torchvision import models

# Modify config if you are conducting different models
from p2_model_cfg import myfcn32_cfg, myfcn8_cfg, resnet50_fcn8_cfg
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
            # # AutoAugment(),
            # # AutoAugment(AutoAugmentPolicy.CIFAR10),
            # # transforms.Resize(256),
            # transforms.RandomRotation(15),
            # transforms.RandomHorizontalFlip(p=0.5),
            # # transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0),
            # transforms.RandomAffine(0, None, (0.7, 1.3)),
            # # ToTensor is needed to convert the type, PIL IMG,  to the typ, float tensor.  
            transforms.ToTensor(),
            
            # experimental normalization for image classification 
            transforms.Normalize(means, stds),
        ])

val_tfm = transforms.Compose([
            # transforms.Resize(256),
            transforms.ToTensor(),
            transforms.Normalize(means, stds),
        ])


unlabel_tfm = transforms.RandomChoice( 
    [
        transforms.Compose([
            transforms.RandomHorizontalFlip(),
            # transforms.ColorJitter(0.1, 0.1),
            transforms.RandomAffine(0, None, (0.8, 1.2)),
            transforms.ToTensor(),
        ]),
        
        transforms.Compose([
            transforms.RandomHorizontalFlip(),
            AutoAugment(),
            transforms.RandomAffine(0, None, (0.8, 1.2)),
            transforms.ToTensor(),
        ])
    ]
)
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
    fixed_seed(1)
    

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

    if cfg['model_type'] == 'fcn32':
        model = myFCN32s(num_out=7)
    elif cfg['model_type'] == 'fcn8':
        model = myFCN8s(num_out=7)
    elif cfg['model_type'] == 'resnet50_fcn8':
        model = myResnet50FCN8s(num_out=7)
    # elif cfg['model_type'] == 'pretrained_resnet50':
    #     model = models.resnet50(weights='ResNet50_Weights.DEFAULT')
    #     model.fc = nn.Sequential(nn.Linear(2048, num_out))

    # print model's architecture
    print(model)
    # print(cfg)
    # Get your training Data 
    ## TO DO ##
    # You need to define your cifar10_dataset yourself to get images and labels for earch data
    # Check myDatasets.py 
      
    # train_set = CustomImageDataset(data_folder_path=training_data_path, have_label=True, transform=train_tfm)
    # val_set = CustomImageDataset(data_folder_path=val_data_path, have_label=True, transform=val_tfm)
    train_set = CustomImageDataset(data_folder_path=training_data_path, have_label=True, transform=train_tfm)
    val_set = CustomImageDataset(data_folder_path=val_data_path, have_label=True, transform=val_tfm)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    if do_semi:
        unlabeled_set_1 = get_cifar10_unlabeled_set("./p2_data/unlabeled")  
        # unlabeled_loader = DataLoader(unlabeled_set_1, batch_size=batch_size, shuffle=True)
        unlabeled_set_2 = get_cifar10_unlabeled_set("./p2_data/public_test")  
        unlabeled_set = ConcatDataset([unlabeled_set_1, unlabeled_set_2])
        unlabeled_loader = DataLoader(unlabeled_set, batch_size=batch_size, shuffle=True)
    else:
        unlabeled_set = None
    
    # define your loss function and optimizer to unpdate the model's parameters.
    optimizer = getattr(torch.optim, cfg['optimizer'])(model.parameters(), **cfg['optim_hparas'])
    ### optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-6, nesterov=True)
    ### optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    # scheduler = None
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

    




    # start_epoch = 0
    # # # step 5: move Net to GPU if available
    # # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # # print('Using device:', device)
    # # net.to(device)

    # # step 6: main loop
    # for epoch in range(start_epoch, start_epoch + num_epoch):
    #     model.train()
    #     running_loss = 0.0
    #     # for i, data in enumerate(train_loader):
    #     for batch_idx, (data, label, sat_path, mask_path,) in enumerate(tqdm(train_loader)):
    #         # inputs, labels = data[0].to(device), data[1].to(device)
    #         data = data.to(device)
    #         label = label.to(device)


    #         # zero the parameter gradients
    #         optimizer.zero_grad()

    #         # forward + backward + optimize
    #         outputs = model(data)
    #         loss = criterion(outputs, label)
    #         loss.backward()
    #         optimizer.step()
    #         # print statistics
    #         running_loss += loss.item()
    #         prefix = 'Epoch [{}/{}]-'.format(epoch + 1, start_epoch + num_epoch)
    #         # if (batch_idx + 1) % 10 == 0:  # print every 10 mini-batches
    #         #     suffix = 'Train Loss: {:.4f}'.format(running_loss / (i + 1))
    #         #     progress_bar(batch_idx + 1, len(train_loader), prefix, suffix)
    #         # if configs.test_run:
    #         #     break

    #     # print Valid mIoU per epoch
    #     model.eval()
    #     with torch.no_grad():
    #         val_metrics = IOU()
    #         # for val_data in val_loader:
    #         for batch_idx, (data, label, sat_path, mask_path,) in enumerate(tqdm(val_loader)):
    #             images, labels, img_fn_prefixs = data[0].to(device), val_data[1], val_data[2]
    #             outputs = model(images)
    #             predicted = torch.argmax(outputs, dim=1).cpu().numpy()

    #             val_metrics.batch_iou(predicted, labels.cpu().numpy())

    #             # save predicted mask png file
    #             save_mask(configs.p2_output_dir, predicted, img_fn_prefixs)

    #         print('\nValid mIoU (me): {}'
    #               .format(val_metrics.miou()))

    #         # TA's mIoU:
    #         # print('')
    #         pred = read_masks(configs.p2_output_dir)
    #         labels = read_masks('/home/hsien/dlcv/hw1-ehsienmu/hw1_data/hw1_data/p2_data/validation')
    #         miou = mean_iou_score(pred, labels)

    #         if pre_val_miou < miou:
    #             checkpoint = {
    #                 'net': net.state_dict(),
    #                 'epoch': epoch,
    #                 'optim': optimizer.state_dict(),
    #                 'uid': uid,
    #                 'miou': miou
    #             }
    #             save_checkpoint(checkpoint,
    #                             os.path.join(configs.ckpt_path, "Vgg16FCN8-{}.pt".format(uid[:8])))
    #             # pre_val_miou = val_metrics.mean_iou
    #             pre_val_miou = miou
    #             best_epoch = epoch + 1

    #         # report
    #         rpt_images, rpt_labels, rpt_fn_prefix = report_batch[0].to(device), report_batch[1], report_batch[2]
    #         rpt_outputs = net(rpt_images)
    #         rpt_pred = torch.argmax(rpt_outputs, dim=1).cpu().numpy()
    #         rpt_fn = ["{}-{}-epochs_{}".format(uid[:8], f, epoch) for f in rpt_fn_prefix]
    #         save_mask('./test_miou', rpt_pred, rpt_fn)












if __name__ == '__main__':
    train_interface(cfg = resnet50_fcn8_cfg)

