# PyTorch libraries and modules
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout

# Creating a CNN class
import torch.nn as nn
import torchvision.models as models

from data_vis import show_image

def channel_size(input_size, kernel_size, stride, padding):
    return 1 + (input_size + 2 * padding - kernel_size) // stride

# Creating a CNN class
class ConvNeuralNet(nn.Module):
    #  Determine what layers and their order in CNN object 
    def __init__(self, img_size, k_size, dropout_val):
        super(ConvNeuralNet, self).__init__()

        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)

        self.conv_layer1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=7)
        s = channel_size(img_size, 7, 1, 0)
        # self.max_pool1 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        s = channel_size(s, 2, 2, 0)

        self.conv_layer2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5)
        s = channel_size(s, 5, 1, 0)
        # self.max_pool1 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        s = channel_size(s, 2, 2, 0)
        
        self.conv_layer3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        s = channel_size(s, 3, 1, 0)

        self.conv_layer4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
        s = channel_size(s, 3, 1, 0)
        # self.max_pool2 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        # s = channel_size(s, 2, 2, 0)
        
        self.conv_layer5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)
        s = channel_size(s, 3, 1, 0)
        # self.conv_layer6 = nn.Conv2d(in_channels=64, out_channels=final_channels, kernel_size=k_size)
        # s = channel_size(s, k_size, 1, 0)
        # self.max_pool3 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        s = channel_size(s, 2, 2, 0)

        self.fc1 = nn.Linear(s**2 * 128, 128)
        # self.fc2 = nn.Linear(128, 128)
        # self.fc3 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, 1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_val)
    
    # Progresses data across layers    
    def forward(self, x):
        out = self.conv_layer1(x)
        out = self.relu(out)
        # out = self.dropout(out)
        out = self.pool(out)

        out = self.conv_layer2(out)
        out = self.relu(out)
        # out = self.dropout(out)
        out = self.pool(out)
        
        out = self.conv_layer3(out)
        out = self.relu(out)
        # out = self.dropout(out)

        out = self.conv_layer4(out)
        out = self.relu(out)
        # out = self.dropout(out)

        out = self.conv_layer5(out)
        out = self.relu(out)
        # out = self.dropout(out)
        out = self.pool(out)

        # out = self.conv_layer6(out)
        # out = self.relu(out)
        # out = self.dropout(out)
        # out = self.max_pool3(out)
                
        out = out.reshape(out.size(0), -1)
        
        out = self.fc1(out)
        out = self.relu(out)
        # out = self.dropout(out)

        out = self.fc2(out)
        # out = self.relu(out)
        # out = self.dropout(out)

        # out = self.fc3(out)
        # out = self.relu(out)
        # out = self.dropout(out)

        # out = self.fc4(out)
        return out
