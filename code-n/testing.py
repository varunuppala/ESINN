from model import ConvNeuralNet
import torch
from torch.optim import Adam, SGD
from torch.nn import Linear, ReLU, CrossEntropyLoss,MSELoss
from model import ConvNeuralNet
import time

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

from torch import nn, cuda, optim
import torch.nn.functional as F

import os

#device = ("cuda" if torch.cuda.is_available() else "cpu")

device = ("mps" if torch.backends.mps.is_built() else "cpu")


def validateModel(model, valLoader):
    '''
    Overview:
        Predicts a validation set using the CNN model and prints accuracy values.
        
    Inputs:
        model - Machine learning model.
        valLoader - Validation set.
    '''
    valLoss = 0
    test_count = 0
    test_acc = 0
    test_loss = 0
    
    # Confusion matrix lists
    pred = []
    true = []
    d = {}
    
    criterion = CrossEntropyLoss()
    
    print('Model Validation.')
    
    model.eval()
    
    with torch.no_grad():
        for inputs, labels in valLoader:
            
            # Calculates loss value.
            inputs, labels = inputs.to(device), labels.to(device)
            #labels = labels - 1
            output = model.forward(inputs)
            valLoss = criterion(output, labels)

            test_count += len(inputs)
            test_loss += valLoss.data.item()
            _,pred = torch.max(output.data,1)
            test_acc += int(torch.sum(pred == labels.data))

            for i,j in zip(pred,labels.data):
            	diff = abs(i.item() - j.item())
            	if diff not in d:
            		d[diff] = 1
            	else:
            		d[diff] += 1
            
    print(d)
    
    #print('Loss: {:.6f} \t Top-1 Accuracy: {:.6f} \t Top-5 Accuracy: {:.6f}\n'.format(valLoss, val1Acc, val5Acc))
    
    print("Test acc:",test_acc*100/test_count)