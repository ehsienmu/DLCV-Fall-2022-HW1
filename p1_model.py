import torch
import torch.nn as nn

PRINT_SHAPE = False


class residual_block(nn.Module):
    def __init__(self, in_channels):
        super(residual_block, self).__init__()
        
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(in_channels))

        self.relu = nn.ReLU()
        
    def forward(self,x):
        ## TO DO ## 
        # Perform residaul network. 
        # You can refer to our ppt to build the block. It's ok if you want to do much more complicated one. 
        # i.e. pass identity to final result before activation function 
        # pass
        residual = x
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv1(x)
        x += residual
        x = self.relu(x)
        return x


class myResnet(nn.Module):
    def __init__(self, in_channels=3, num_out=50):
        super(myResnet, self).__init__()
        
        self.stem_conv = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=128, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(128))

        self.res1 = residual_block(in_channels=128)
        self.myconv1 = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2), # 16,128,15,15
                             nn.BatchNorm2d(256),
                             nn.MaxPool2d(kernel_size=3, stride=2),
                             nn.ReLU(),
                             )
        self.res2 = residual_block(in_channels=256) 
        self.myconv2 = nn.Sequential(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2),
                             nn.BatchNorm2d(512),
                             nn.MaxPool2d(kernel_size=3, stride=2),
                             nn.ReLU(),
                             )
        
        self.res3 = residual_block(in_channels=512)
        
        self.fc1 = nn.Sequential(nn.Linear(512, 256), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(256,128), nn.ReLU())
        self.fc3 = nn.Linear(128, num_out)

        self.activation = nn.ReLU()
        # pass
        # self.tempconv1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2)
        # self.batchnorm1 = nn.BatchNorm2d(128)
        # self.maxpool = nn.MaxPool2d
    def forward(self,x):

        x = self.stem_conv(x)
        x = self.activation(x)
        # print('after self.activation(x):', x.shape) # torch.Size([16, 64, 32, 32])
        x = self.res1(x)
        # print('after self.res1(x):', x.shape) # torch.Size([16, 64, 32, 32])
        x = self.myconv1(x)
        # print('after self.myconv1(x):', x.shape) 
        x = self.res2(x)
        # print('after self.res2(x):', x.shape) 
        x = self.myconv2(x)
        x = self.res3(x)
        # print('after self.myconv2(x):', x.shape) 
        x = torch.flatten(x, start_dim=1, end_dim=-1)
        
        # x = x.flatten(x)
        # print(x.shape)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)        
        out = x
        return out

class myCNN(nn.Module):

    def __init__(self, num_out=10):
        super(myCNN, self).__init__()
        # input_shape = (1, 32, 32)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=0), 
            # output_shape=(16, 30, 30) 
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2),
        )
        # input_shape = (16, 30, 30) 
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 0), # output_shape=(32, 28, 28)
            nn.ReLU(),
            # nn.MaxPool2d(2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 256, 3, 1, 0), # output_shape=(64, 26, 26)
            nn.ReLU(),
            # nn.MaxPool2d(2),
        )
        self.conv4 = nn.Sequential( # output_shape=(128, 24, 24)
            nn.Conv2d(256, 512, 3, 1, 0),
            nn.ReLU(),
            # nn.MaxPool2d(2),
        )
        self.conv5 = nn.Sequential( # output_shape=(256, 22, 22)
            nn.Conv2d(512, 32, 3, 1, 0),
            nn.ReLU(),
            # nn.MaxPool2d(2),
        )
        self.fc = nn.Linear(32 * 22 * 22, num_out)
        # self.conv_layers = [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5]

    def forward(self, x):
        if PRINT_SHAPE:
            print('input: ', x.shape)
        x = self.conv1(x)
        if PRINT_SHAPE:
            print('after conv1: ', x.shape)
        x = self.conv2(x)
        if PRINT_SHAPE:
            print('after conv2: ', x.shape)
        x = self.conv3(x)
        if PRINT_SHAPE:
            print('after conv3: ', x.shape)
        x = self.conv4(x)
        if PRINT_SHAPE:
            print('after conv4: ', x.shape)
        x = self.conv5(x)
        if PRINT_SHAPE:
            print('after conv5: ', x.shape)
        # flatten the output of conv to (batch_size, 512 * 8 * 8)
        x = x.view(x.size(0), -1)       
        output = self.fc(x)
        return output#, x    # return x for visualization
    
    def sec_last_out(self, x):
        if PRINT_SHAPE:
            print('input: ', x.shape)
        x = self.conv1(x)
        if PRINT_SHAPE:
            print('after conv1: ', x.shape)
        x = self.conv2(x)
        if PRINT_SHAPE:
            print('after conv2: ', x.shape)
        x = self.conv3(x)
        if PRINT_SHAPE:
            print('after conv3: ', x.shape)
        x = self.conv4(x)
        if PRINT_SHAPE:
            print('after conv4: ', x.shape)
        x = self.conv5(x)
        if PRINT_SHAPE:
            print('after conv5: ', x.shape)
        # flatten the output of conv to (batch_size, 512 * 8 * 8)
        x = x.view(x.size(0), -1)       
        # output = self.fc(x)
        return x