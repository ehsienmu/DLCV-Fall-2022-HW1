# from re import X
import torch
import torch.nn as nn
from torchvision import models

PRINT_SHAPE = False


class myFCN32s(nn.Module):
    def __init__(self, num_out=1000):
        super(myFCN32s, self).__init__()
        
        self.encoder = models.vgg16(weights='VGG16_Weights.DEFAULT').features
        self.convs = nn.Sequential(nn.Conv2d(512, 1028, 3, 1, 1),
                                nn.ReLU(),
                                # nn.Dropout(),
                                nn.Conv2d(1028, 1028, 2, 1),
                                nn.ReLU(),
                                # nn.Dropout(),
                                nn.Conv2d(1028, num_out, 1),
                                # nn.ReLU(inplace=True),
                                # nn.Dropout(),
                                )
        self.upsampling = nn.ConvTranspose2d(num_out, num_out, kernel_size=64, stride=32)#, padding=12)
        # self.vgg = models.vgg16(weights='VGG16_Weights.DEFAULT')
        for param in self.encoder.parameters():
            param.requires_grad = False
            
    def forward(self, x):
        x = self.encoder(x)  
        x = self.convs(x)
        x = self.upsampling(x)
        return x



class myFCN8s(nn.Module):
    def __init__(self, num_out=7):
        super(myFCN8s, self).__init__()
        # self.encoder = models.vgg16(weights='VGG16_Weights.DEFAULT').features
        self.convs = nn.Sequential(nn.Conv2d(512, 1028, 3, 1, 1),
                                nn.ReLU(),
                                # nn.Dropout(),
                                nn.Conv2d(1028, 1028, 2, 1),
                                nn.ReLU(),
                                # nn.Dropout(),
                                nn.Conv2d(1028, num_out, 1),
                                # nn.ReLU(inplace=True),
                                # nn.Dropout(),
                                )
        # self.vgg = models.vgg16(weights='VGG16_Weights.DEFAULT')
        self.vgg_pool2 = nn.Sequential(*[models.vgg16(weights='VGG16_Weights.DEFAULT').features[i] for i in range(0, 5)])
        self.vgg_pool4 = nn.Sequential(*[models.vgg16(weights='VGG16_Weights.DEFAULT').features[i] for i in range(5, 10)])
        self.vgg_pool8 = nn.Sequential(*[models.vgg16(weights='VGG16_Weights.DEFAULT').features[i] for i in range(10, 17)])
        self.vgg_pool16 = nn.Sequential(*[models.vgg16(weights='VGG16_Weights.DEFAULT').features[i] for i in range(17, 24)])
        self.vgg_pool32 = nn.Sequential(*[models.vgg16(weights='VGG16_Weights.DEFAULT').features[i] for i in range(24, 31)])
        for param in self.vgg_pool2.parameters():
            param.requires_grad = False
        for param in self.vgg_pool4.parameters():
            param.requires_grad = False
        for param in self.vgg_pool8.parameters():
            param.requires_grad = False
        for param in self.vgg_pool16.parameters():
            param.requires_grad = False
        for param in self.vgg_pool32.parameters():
            param.requires_grad = False
        self.upsampling2 = nn.ConvTranspose2d(num_out, num_out, kernel_size=4, stride=2)#, padding=12)
        self.upsampling4 = nn.ConvTranspose2d(num_out, num_out, kernel_size=8, stride=4)#, padding=12)
        self.upsampling8 = nn.ConvTranspose2d(num_out, num_out, kernel_size=16, stride=8)
        self.upsampling16 = nn.ConvTranspose2d(num_out, num_out, kernel_size=32, stride=16)#, padding=12)
        self.upsampling32 = nn.ConvTranspose2d(num_out, num_out, kernel_size=64, stride=32)#, padding=12)
        
        self.convs_channel512 = nn.Conv2d(512, num_out, kernel_size=1)
        self.convs_channel256 = nn.Conv2d(256, num_out, kernel_size=1)
    def forward(self, x):
        if PRINT_SHAPE:
            print('x.size():', x.size())
        x1_2 = self.vgg_pool2(x)  #1/2
        if PRINT_SHAPE:
            print('x1_2.size():', x1_2.size())
        x1_4 = self.vgg_pool4(x1_2)  #1/4
        if PRINT_SHAPE:
            print('x1_4.size():', x1_4.size())
        x1_8 = self.vgg_pool8(x1_4)  #1/8
        if PRINT_SHAPE:
            print('x1_8.size():', x1_8.size())
        x1_16 = self.vgg_pool16(x1_8)  #1/16
        if PRINT_SHAPE:
            print('x1_16.size():', x1_16.size())
        x1_32 = self.vgg_pool32(x1_16)  #1/32
        if PRINT_SHAPE:
            print('before convs x1_32.size():', x1_32.size())
        x1_32 = self.convs(x1_32)
        if PRINT_SHAPE:
            print('after conv x1_32.size():', x1_32.size())

        pred1_16 = self.upsampling2(x1_32)
        if PRINT_SHAPE:
            print('pred1_16.size():', pred1_16.size())

        sum_1_16 = pred1_16 + self.convs_channel512(x1_16)
        if PRINT_SHAPE:
            print('sum_1_16.size():', sum_1_16.size())
        # fcn16_out = self.upsampling16(sum_1_16) # 1/16->1

        pred1_8 = self.upsampling2(sum_1_16)#[:, :, 1:-1, 1:-1]
        if PRINT_SHAPE:
            print('pred1_8.size():', pred1_8.size())

        
        sum_1_8 = pred1_8[:, :, 1:-1, 1:-1] + self.convs_channel256(x1_8)
        if PRINT_SHAPE:
            print('sum_1_8.size():', sum_1_8.size())

        fcn8_out = self.upsampling8(sum_1_8)[:, :, 4:-4, 4:-4]
        if PRINT_SHAPE:
            print('fcn8_out.size():', fcn8_out.size())

        # fcn32_out = self.upsampling32(x1_32) # 1/32->1
        return fcn8_out



class myResnet50FCN8s(nn.Module):
    def __init__(self, num_out=7):
        super(myResnet50FCN8s, self).__init__()
        # self.encoder = models.resnet50(weights='ResNet50_Weights.DEFAULT')
        self.convs = nn.Sequential(nn.Conv2d(2048, 1028, 3, 1, 1),
                                nn.ReLU(),
                                # nn.Dropout(),
                                nn.Conv2d(1028, 1028, 2, 1),
                                nn.ReLU(),
                                # nn.Dropout(),
                                nn.Conv2d(1028, num_out, 1),
                                # nn.ReLU(inplace=True),
                                # nn.Dropout(),
                                )
        # self.vgg = models.vgg16(weights='VGG16_Weights.DEFAULT')
        self.res_conv1 = nn.Sequential(*[
            models.resnet50(weights='ResNet50_Weights.DEFAULT').conv1,
            models.resnet50(weights='ResNet50_Weights.DEFAULT').bn1,
            models.resnet50(weights='ResNet50_Weights.DEFAULT').relu,
            models.resnet50(weights='ResNet50_Weights.DEFAULT').maxpool,
            ])
        self.res_layer1 = nn.Sequential(*[
            models.resnet50(weights='ResNet50_Weights.DEFAULT').layer1
            ])
        self.res_layer2 = nn.Sequential(*[
            models.resnet50(weights='ResNet50_Weights.DEFAULT').layer2
            ])
        self.res_layer3 = nn.Sequential(*[
            models.resnet50(weights='ResNet50_Weights.DEFAULT').layer3
            ])
        self.res_layer4 = nn.Sequential(*[
            models.resnet50(weights='ResNet50_Weights.DEFAULT').layer4
            ])
        # self.vgg_pool2 = nn.Sequential(*[ models.resnet50(weights='ResNet50_Weights.DEFAULT')[i] for i in range(0, 5)])
        # self.vgg_pool4 = nn.Sequential(*[models.vgg16(weights='VGG16_Weights.DEFAULT').features[i] for i in range(5, 10)])
        # self.vgg_pool8 = nn.Sequential(*[models.vgg16(weights='VGG16_Weights.DEFAULT').features[i] for i in range(10, 17)])
        # self.vgg_pool16 = nn.Sequential(*[models.vgg16(weights='VGG16_Weights.DEFAULT').features[i] for i in range(17, 24)])
        # self.vgg_pool32 = nn.Sequential(*[models.vgg16(weights='VGG16_Weights.DEFAULT').features[i] for i in range(24, 31)])
        # for param in self.vgg_pool2.parameters():
        #     param.requires_grad = False
        # for param in self.vgg_pool4.parameters():
        #     param.requires_grad = False
        # for param in self.vgg_pool8.parameters():
        #     param.requires_grad = False
        # for param in self.vgg_pool16.parameters():
        #     param.requires_grad = False
        # for param in self.vgg_pool32.parameters():
        #     param.requires_grad = False
        self.upsampling2 = nn.ConvTranspose2d(num_out, num_out, kernel_size=4, stride=2)#, padding=12)
        self.upsampling4 = nn.ConvTranspose2d(num_out, num_out, kernel_size=8, stride=4)#, padding=12)
        self.upsampling8 = nn.ConvTranspose2d(num_out, num_out, kernel_size=16, stride=8)
        self.upsampling16 = nn.ConvTranspose2d(num_out, num_out, kernel_size=32, stride=16)#, padding=12)
        self.upsampling32 = nn.ConvTranspose2d(num_out, num_out, kernel_size=64, stride=32)#, padding=12)
        
        self.convs_channel1024 = nn.Conv2d(1024, num_out, kernel_size=1)
        self.convs_channel512 = nn.Conv2d(512, num_out, kernel_size=1)
        self.convs_channel256 = nn.Conv2d(256, num_out, kernel_size=1)
    def forward(self, x):
        if PRINT_SHAPE:
            print('x.size():', x.size())
        x1_2 = self.res_conv1(x)  #1/2
        if PRINT_SHAPE:
            print('x1_2.size():', x1_2.size())
        x1_4 = self.res_layer1(x1_2)  #1/4
        if PRINT_SHAPE:
            print('x1_4.size():', x1_4.size())
        x1_8 = self.res_layer2(x1_4)  #1/8
        if PRINT_SHAPE:
            print('x1_8.size():', x1_8.size())
        x1_16 = self.res_layer3(x1_8)  #1/16
        if PRINT_SHAPE:
            print('x1_16.size():', x1_16.size())
        x1_32 = self.res_layer4(x1_16)  #1/32
        if PRINT_SHAPE:
            print('before conv x1_32.size():', x1_32.size())
        x1_32 = self.convs(x1_32)
        if PRINT_SHAPE:
            print('after conv x1_32.size():', x1_32.size())

        pred1_16 = self.upsampling2(x1_32)
        if PRINT_SHAPE:
            print('pred1_16.size():', pred1_16.size())

        sum_1_16 = pred1_16 + self.convs_channel1024(x1_16)
        if PRINT_SHAPE:
            print('sum_1_16.size():', sum_1_16.size())
        # fcn16_out = self.upsampling16(sum_1_16) # 1/16->1

        pred1_8 = self.upsampling2(sum_1_16)#[:, :, 1:-1, 1:-1]
        if PRINT_SHAPE:
            print('pred1_8.size():', pred1_8.size())

        
        sum_1_8 = pred1_8[:, :, 1:-1, 1:-1] + self.convs_channel512(x1_8)
        if PRINT_SHAPE:
            print('sum_1_8.size():', sum_1_8.size())

        x = self.upsampling8(sum_1_8)[:, :, 4:-4, 4:-4]
        if PRINT_SHAPE:
            print('x.size():', x.size())

        # fcn32_out = self.upsampling32(x1_32) # 1/32->1
        return x


