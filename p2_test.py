import torch
import os

from torch.utils.data import DataLoader, ConcatDataset
import torch.optim as optim 
import torch.nn as nn
from tqdm import tqdm

from p2_model import *
from p2_datasets import *
# from p2_tool import fixed_seed
# from torchvision import models

# Modify config if you are conducting different models
from p2_model_cfg import myfcn32_cfg, myfcn8_cfg, resnet50_fcn8_cfg
import argparse


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


# def write_masks(pred_masks, output_path):
#     # each pred is (1, 512, 512)
#     os.makedirs(output_path, exist_ok=True)
#     image_cnt, h, w = pred_masks.shape
#     all_masks_color = np.empty((image_cnt, h, w, 3))
#     for index, pred_class in enumerate(pred_masks):
#         all_masks_color[index, pred_class == 0] = [0, 255, 255]
#         all_masks_color[index, pred_class == 1] = [255, 255, 0]
#         all_masks_color[index, pred_class == 2] = [255, 0, 255]
#         all_masks_color[index, pred_class == 3] = [0, 255, 0]
#         all_masks_color[index, pred_class == 4] = [0, 0, 255]
#         all_masks_color[index, pred_class == 5] = [255, 255, 255]
#         all_masks_color[index, pred_class == 6] = [0, 0, 0]
#     for index, img in enumerate(all_masks_color):
#         imageio.imwrite(os.path.join(output_path, format(index, '04d') + ".png"), img.astype(np.uint8))

def write_single_mask(pred_mask, output_path, img_index_str):
    # each pred is (1, 512, 512)
    os.makedirs(output_path, exist_ok=True)
    pred_mask = pred_mask.cpu().detach().numpy()
    # for i in range(7):
        
    #     int(np.sum(pred_mask == i))
        
    # print(pred_mask.shape)
    
    _, h, w = pred_mask.shape
    masks_color = np.empty((h, w, 3))
    
    masks_color[pred_mask[0] == 0] = [0, 255, 255]
    masks_color[pred_mask[0] == 1] = [255, 255, 0]
    masks_color[pred_mask[0] == 2] = [255, 0, 255]
    masks_color[pred_mask[0] == 3] = [0, 255, 0]
    masks_color[pred_mask[0] == 4] = [0, 0, 255]
    masks_color[pred_mask[0] == 5] = [255, 255, 255]
    masks_color[pred_mask[0] == 6] = [0, 0, 0]
    
    imageio.imwrite(os.path.join(output_path, img_index_str + ".png"), masks_color.astype(np.uint8))


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
        if have_label:
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
        sat = Image.open(self.sats[idx]).convert("RGB")
        
        if self.masks != None:
            seg = imageio.imread(self.masks[idx])
            mask = torch.tensor(read_masks(seg, sat.size))
        # print('type(mask):', type(mask)) # <class 'torch.Tensor'>   
        if self.transform != None:   
            sat = self.transform(sat)
            
        # print('type(sat):', type(sat)) #<class 'PIL.Image.Image'> 

        if self.masks != None:
            #  print(type((transform_img, self.labels[idx])))
            return (sat, mask.long(), self.sats[idx], self.masks[idx])
        else:
            return (sat, self.sats[idx])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--input_dir', default='', type=str)
    parser.add_argument('--output_dir', default='', type=str)
    parser.add_argument('--model_file', default='', type=str)

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # fixed random seed
    fixed_seed(1)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    
    """ training hyperparameter """

    model = myResnet50FCN8s(num_out=7)

    load_parameters(model, args.model_file)
    # Put model's parameters on your device
    model = model.to(device)
    
    # print(model)
   
    # train_set = CustomImageDataset(data_folder_path=training_data_path, have_label=True, transform=train_tfm)
    val_set = CustomImageDataset(data_folder_path=args.input_dir, have_label=False, transform=val_tfm)
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
    with torch.no_grad():
        model.eval()
        val_loss = 0.0
        corr_num = 0
        val_acc = 0.0
        
        ## TO DO ## 
        # Finish forward part in validation. You can refer to the training part 
        # Note : You don't have to update parameters this part. Just Calculate/record the accuracy and loss. 

        # for batch_idx, (data, fname,) in enumerate(tqdm(val_loader)):
        for batch_idx, (data, fname,) in enumerate(tqdm(val_loader)):
            # put the data and label on the device
            # note size of data (B,C,H,W) --> B is the batch size
            data = data.to(device)
            # label = label.to(device)

            # pass forward function define in the model and get output 
            output = model(data) 

            # calculate the loss between output and ground truth
            # loss = criterion(output, label)
            
            # # discard the gradient left from former iteration 
            # optimizer.zero_grad()

            # # calcualte the gradient from the loss function 
            # loss.backward()
            
            # # if the gradient is too large, we dont adopt it
            # grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm= 5.)
            
            # # Update the parameters according to the gradient we calculated
            # optimizer.step()

            # val_loss += loss.item()

            # predict the label from the last layers' output. Choose index with the biggest probability 
            pred = output.argmax(dim=1)
            
            img_index_str = fname[0].split('/')[-1][:4]

            write_single_mask(pred, args.output_dir, img_index_str)
            # correct if label == predict_label
            # corr_num += (pred.eq(label.view_as(pred)).sum().item())

        # scheduler += 1 for adjusting learning rate later
        
        # averaging training_loss and calculate accuracy
        # val_loss = val_loss / len(val_loader.dataset) 
        # val_acc = corr_num / len(val_loader.dataset)
        
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
        