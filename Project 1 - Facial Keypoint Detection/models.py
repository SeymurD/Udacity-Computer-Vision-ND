## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I
from torchvision import models

from collections import OrderedDict

# *** Conv2d output dimensions ***
# height_out = (height_in + 2*padding - dilation*(kernel_size - 1) - 1)/stride + 1
# width_out = (width_in + 2*padding - dilation*(kernel_size - 1) - 1)/stride + 1
# weights_out = height_out * width_out * channels_out
#
# With values: strid = 1, padding = 0, dilation = 1
# height_out = height_in - kernel_size + 1
# width_out = width_in - kernel_size + 1
#
# *** MaxPool2d output dimensions ***
# height_out = (height_in + 2*padding - dilation*(kernel_size - 1) - 1)/stride + 1
# width_out = (width_in + 2*padding - dilation*(kernel_size - 1) - 1)/stride + 1
# weights_out = height_out * width_out * channels_out
#
# With values: strid = 2, padding = 0, dilation = 1
# height_out = (height_in - kernel_size)/2 + 1
# width_out = (width_in - kernel_size)/2 + 1


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        
        
        # ----------------------------------------------Layer 1-----------------------------------------------------
        ## input images are 224x224 pixels, 3x3 kernel, padding by 1 
        ## output size = (W-F)/S +1 = (224-3+2)/1 +1 = 224
        # the output Tensor for one image, will have the dimensions: (32, 224, 224)
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        # maxpool layer
        # pool with kernel_size=2, stride=2
        self.pool1 = nn.MaxPool2d(2, 2)
        # output tensor (32, 112, 112)
        self.fc_drop1 = nn.Dropout(p=0.2)
        
        # ----------------------------------------------Layer 2-----------------------------------------------------
        ## input images are 112x112 pixels, 3x3 kernel, padding by 1 
        ## output size = (W-F)/S +1 = (112-3+2)/1 +1 = 112
        # the output Tensor for one image, will have the dimensions: (64, 112, 112)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        # pool with kernel_size=2, stride=2
        self.pool2 = nn.MaxPool2d(2, 2)
        # output tensor (64, 56, 56)
        self.fc_drop2 = nn.Dropout(p=0.2)
        
        # ----------------------------------------------Layer 3-----------------------------------------------------
        ## input images are 56x56 pixels, 3x3 kernel, padding by 1 
        ## output size = (W-F)/S +1 = (56-3+2)/1 +1 = 56
        # the output Tensor for one image, will have the dimensions: (128, 56, 56)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        # pool with kernel_size=2, stride=2
        self.pool3 = nn.MaxPool2d(2, 2)
        # output tensor (128, 28, 28)
        self.fc_drop3 = nn.Dropout(p=0.2)
        
        # ----------------------------------------------Layer 4-----------------------------------------------------
        ## input images are 28x28 pixels, 3x3 kernel, padding by 1 
        ## output size = (W-F)/S +1 = (28-3+2)/1 +1 = 28
        # the output Tensor for one image, will have the dimensions: (256, 28, 28)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        # pool with kernel_size=2, stride=2
        self.pool4 = nn.MaxPool2d(2, 2)
        # output tensor (256, 14, 14)
        self.fc_drop4 = nn.Dropout(p=0.2)
        
        # ----------------------------------------------Layer 5-----------------------------------------------------
        ## input images are 14x14 pixels, 3x3 kernel, padding by 1 
        ## output size = (W-F)/S +1 = (14-3+2)/1 +1 = 14
        # the output Tensor for one image, will have the dimensions: (512, 14, 14)
        self.conv5 = nn.Conv2d(256, 512, 3, padding=1)
        # pool with kernel_size=2, stride=2
        # output tensor (512, 7, 7)
        self.pool5 = nn.MaxPool2d(2, 2)
   
        # --------------------------------------Full-Connected Layers-----------------------------------------------
        self.fc6 = nn.Linear(512*7*7, 136)    

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        
        # two conv/relu + pool layers
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.fc_drop1(x)
        
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.fc_drop2(x)
        
        x = self.pool3(F.relu(self.conv3(x)))
        x = self.fc_drop3(x)
        
        x = self.pool4(F.relu(self.conv4(x)))
        x = self.fc_drop4(x)
        
        x = self.pool5(F.relu(self.conv5(x)))

        # prep for linear layer
        # this line of code is the equivalent of Flatten in Keras
        
        x = x.view(x.size(0), -1)
        
        # fully-connected layer
        x = self.fc6(x)

        return x
    
    
    
class NaimishNet(nn.Module):
    def __init__(self, image_size, output_size = 136, kernels = [5,5,5,5],out_channels = [32,64,128,256],
                dropout_p = [0, 0, 0, 0, 0, 0], use_padding=True, use_maxp = True):
        super(NaimishNet, self).__init__() 
        # padding only support odd numbered kernels in this implementation
        self.use_padding = use_padding
        
        # init padding
        if self.use_padding:
            self.padding = [int((k-1)/2) for k in kernels]
        else:
            self.padding = [0,0,0,0]
            
        # Find the size of the last maxp output. 
        last_maxp_size = image_size
        for idx, val in enumerate(kernels):
            if self.use_padding:
                last_maxp_size = last_maxp_size//2
            else:
                last_maxp_size = (last_maxp_size - (val-1))//2
        last_maxp_size = out_channels[3] * last_maxp_size * last_maxp_size

        self.conv1 = nn.Sequential(
            OrderedDict([
            ('conv1', nn.Conv2d(1, out_channels[0], kernel_size=kernels[0], padding=self.padding[0])),
            ('relu1', nn.ReLU())
            ])) # (32, 252, 252)                        
        
        if use_maxp:
            self.maxp1 = nn.Sequential(OrderedDict([
                ('maxp1', nn.MaxPool2d(2, 2)),
                ('dropout1', nn.Dropout(dropout_p[0])),
                ('bachnorm1', nn.BatchNorm2d(out_channels[0]))
                ])) # (32, 126, 126)
        else:
            self.maxp1 = nn.Sequential(OrderedDict([
                ('maxp1', nn.AvgPool2d(2, 2)),
                ('dropout1', nn.Dropout(dropout_p[0])),
                ('bachnorm1', nn.BatchNorm2d(out_channels[0]))
                ])) # (32, 126, 126)

        self.conv2 = nn.Sequential(OrderedDict([
            ('conv2', nn.Conv2d(out_channels[0], out_channels[1], kernel_size=kernels[1], padding=self.padding[1])),
            ('relu2', nn.ReLU())
            ])) # (64, 122, 122)
        
        if use_maxp:
            self.maxp2 = nn.Sequential(OrderedDict([
                ('maxp2', nn.MaxPool2d(2, 2)),
                ('dropout2', nn.Dropout(dropout_p[1])),
                ('bachnorm2', nn.BatchNorm2d(out_channels[1]))
                ])) # (64, 61, 61)
        else:
            self.maxp2 = nn.Sequential(OrderedDict([
                ('maxp2', nn.AvgPool2d(2, 2)),
                ('dropout2', nn.Dropout(dropout_p[1])),
                ('bachnorm2', nn.BatchNorm2d(out_channels[1]))
                ])) # (64, 61, 61)
            
        self.conv3 = nn.Sequential(OrderedDict([
            ('conv3', nn.Conv2d(out_channels[1], out_channels[2], kernel_size=kernels[2], padding=self.padding[2])),
            ('relu3', nn.ReLU())
            ])) # (128, 59, 59)

        if use_maxp:
            self.maxp3 = nn.Sequential(OrderedDict([
                ('maxp3', nn.MaxPool2d(2, 2)),
                ('dropout3', nn.Dropout(dropout_p[2])),
                ('bachnorm3', nn.BatchNorm2d(out_channels[2]))
                ])) # (128, 29, 29)
        else:
            self.maxp3 = nn.Sequential(OrderedDict([
                ('maxp3', nn.AvgPool2d(2, 2)),
                ('dropout3', nn.Dropout(dropout_p[2])),
                ('bachnorm3', nn.BatchNorm2d(out_channels[2]))
                ])) # (128, 29, 29)
            
        self.conv4 = nn.Sequential(OrderedDict([
            ('conv4', nn.Conv2d(out_channels[2], out_channels[3], kernel_size=kernels[3], padding=self.padding[3])),
            ('relu4', nn.ReLU())
            ])) # (256, 27, 27)
        
        if use_maxp:
            self.maxp4 = nn.Sequential(OrderedDict([
                ('maxp4', nn.MaxPool2d(2, 2)),
                ('dropout4', nn.Dropout(dropout_p[3])),
                ('bachnorm4', nn.BatchNorm2d(out_channels[3]))
                ]))  # (256, 13, 13)
        else:
            self.maxp4 = nn.Sequential(OrderedDict([
                ('maxp4', nn.AvgPool2d(2, 2)),
                ('dropout4', nn.Dropout(dropout_p[3])),
                ('bachnorm4', nn.BatchNorm2d(out_channels[3]))
                ]))  # (256, 13, 13)
        
        self.fc1 = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(last_maxp_size, 1024)),
            ('relu5', nn.ReLU()),
            ('dropout5', nn.Dropout(dropout_p[4])),
            ('bachnorm5', nn.BatchNorm1d(1024))
            ])) # (36864, 1024)

        self.fc2 = nn.Sequential(OrderedDict([
            ('fc2', nn.Linear(1024, 1024)),
            ('relu6', nn.ReLU()),
            ('dropout6', nn.Dropout(dropout_p[5])),
            ('bachnorm6', nn.BatchNorm1d(1024))
            ])) # (1024, 1024)

        self.fc3 = nn.Sequential(OrderedDict([
            ('fc3', nn.Linear(1024, output_size))
            ])) # (1024, 136)

    def forward(self, x):
        out = self.conv1(x)
        out = self.maxp1(out)
        out = self.conv2(out)
        out = self.maxp2(out)
        out = self.conv3(out)
        out = self.maxp3(out)
        out = self.conv4(out)
        out = self.maxp4(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out
    
    def __str__(self):
        pretty_net_str = ''
        for layer_name in self._modules:
            pretty_net_str += f'{layer_name}:\n'
            for items in getattr(self, layer_name):
                pretty_net_str += f'{items}\n'
            pretty_net_str += '\n'
        return pretty_net_str
