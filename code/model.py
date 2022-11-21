# PyTorch libraries and modules
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout

# Creating a CNN class
import torch.nn as nn
import torchvision.models as models

def channel_size(input_size, kernel_size, stride, padding):
    return 1 + (input_size + 2 * padding - kernel_size) // stride

# Creating a CNN class
class ConvNeuralNet(nn.Module):
    #  Determine what layers and their order in CNN object 
    def __init__(self, img_size):
        super(ConvNeuralNet, self).__init__()
        k_size = 5
        self.conv_layer1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=k_size)
        s = channel_size(img_size, k_size, 1, 0)
        self.conv_layer2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=k_size)
        s = channel_size(s, k_size, 1, 0)
        self.max_pool1 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        s = channel_size(s, 2, 2, 0)
        
        self.conv_layer3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=k_size)
        s = channel_size(s, k_size, 1, 0)
        final_channels = 64
        self.conv_layer4 = nn.Conv2d(in_channels=64, out_channels=final_channels, kernel_size=k_size)
        s = channel_size(s, k_size, 1, 0)
        # self.max_pool2 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        # s = channel_size(s, 2, 2, 0)
        
        self.fc1 = nn.Linear(s**2 * final_channels, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 1)
    
    # Progresses data across layers    
    def forward(self, x):
        out = self.conv_layer1(x)
        out = self.conv_layer2(out)
        out = self.max_pool1(out)
        
        out = self.conv_layer3(out)
        out = self.conv_layer4(out)
        # out = self.max_pool2(out)
                
        out = out.reshape(out.size(0), -1)
        
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        return out
