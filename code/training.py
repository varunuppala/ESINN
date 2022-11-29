import torch
from torch.optim import Adam, SGD
from torch.nn import Linear, ReLU, CrossEntropyLoss,MSELoss
from model import ConvNeuralNet

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

from torch import nn, cuda, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

import wandb
import numpy as np

from data_vis import show_image


device = ("cuda" if torch.cuda.is_available() else "cpu")

#device = ("mps" if torch.backends.mps.is_built() else "cpu")

#ref : https://www.kaggle.com/code/liamvinson/transfer-learning-cnn

def accuracy(output, labels):
	# output = torch.round(output)
	# # print('IN ACC:')
	# # print('dtype:', output.dtype)
	# equals = output == labels
	# return torch.mean(equals.type(torch.FloatTensor)).item()
	# print(output)
	# print(torch.mean(labels.type(torch.FloatTensor)))
	return torch.sum((torch.abs(output - labels) <= 0.5).type(torch.FloatTensor)).item()


def train(train_loader, val_loader):
	
	num_epochs = 1000

	model = ConvNeuralNet(256, k_size=5, dropout_val=0.0)
	if device == "cuda":
		model = model.cuda()
	print(model)

	optimizer = Adam(model.parameters(), lr=1e-5)
	criterion = nn.MSELoss(reduction='sum')
	
	# We use the pre-defined number of epochs to determine how many iterations to train the network on
	for epoch in range(num_epochs):

		model, avg_train_loss, train_acc, avg_val_loss, val_acc = train_one_epoch(model, train_loader, val_loader, optimizer, criterion)

		print('Epoch [{}/{}], TrainLoss: {:.4f}'.format(epoch+1, num_epochs, avg_train_loss))
		print('\tTrain Loss: {:.4f}\tTrain Accuracy: {:.4f}\tVal Loss: {:.4f}\tVal Accuracy: {:.4f}'.format(avg_train_loss, train_acc, avg_val_loss, val_acc))

	return model


def validateModel(model, valLoader):
    '''
    Overview:
        Predicts a validation set using the CNN model and prints accuracy values.
        
    Inputs:
        model - Machine learning model.
        valLoader - Validation set.
    '''
    valLoss = 0
    val1Acc = 0
    val5Acc = 0
    
    # Confusion matrix lists
    pred = []
    true = []
    
    criterion = nn.NLLLoss()
    
    print('Model Validation.')
    
    model.eval()
    with torch.no_grad():
        for inputs, labels in valLoader:
            
            # Calculates loss value.
            inputs, labels = inputs.to(device), labels.to(device)
            output = model.forward(inputs)
            valLoss += criterion(output, labels).item()

            val1Acc += accuracy(output, labels)
            # # Calculates validation top 1 accuracy.
            # output = torch.exp(output)
            # _, top_class = output.topk(1, dim=1)
            # equals = top_class == labels.view(*top_class.shape)
            # val1Acc += torch.mean(equals.type(torch.FloatTensor)).item()
            
            true += labels.tolist()
            pred += top_class.flatten().tolist()
            
            # Calculates validation top 5 accuracy.
            _, top_class = output.topk(5, dim=1)
            equals = top_class == labels.view(labels.shape[0], 1)
            val5Acc += torch.sum(equals).item() / labels.shape[0]
            
    # Calculates average.
    n = len(valLoader)
    valLoss /= n
    val1Acc /= n
    val5Acc /= n
    
    print('Loss: {:.6f} \t Top-1 Accuracy: {:.6f} \t Top-5 Accuracy: {:.6f}\n'.format(valLoss, val1Acc, val5Acc))
    
    # Confusion matrix
    cm = confusion_matrix(true, pred)
    plt.figure(figsize=(12, 9))
    plt.title('Confusion Matrix')
    sns.heatmap(cm)
    plt.show()


def train_one_epoch(model, train_loader, val_loader, optimizer, criterion):
	model.train()
	
	train_loss = 0
	train_correct = 0
	train_pts = 0

	#Load in the data in batches using the train_loader object
	for i, (images, labels) in enumerate(train_loader):  
		
		# Move tensors to the configured device
		images = images.to(device)
		# labels = labels.type(torch.FloatTensor)
		labels = labels.to(device)
		
		# Forward pass
		outputs = model(images)
		outputs = torch.squeeze(outputs, 1)
		loss = criterion(outputs, labels)

		train_loss += loss.item()
		train_correct += accuracy(outputs, labels)
		train_pts += len(outputs)
		
		# Backward and optimize
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

	avg_train_loss = train_loss / train_pts
	train_acc = train_correct / train_pts

	# Validating the model
	model.eval()

	val_loss = 0
	val_correct = 0
	val_pts = 0

	with torch.no_grad():
		for images, labels in val_loader:
			images, labels = images.to(device), labels.to(device)
			
			outputs = model(images)
			outputs = torch.squeeze(outputs, 1)
			loss = criterion(outputs, labels)

			val_loss += loss.item()
			val_correct += accuracy(outputs, labels)
			val_pts += len(outputs)

	avg_val_loss = val_loss / val_pts
	val_acc = val_correct / val_pts

	return model, avg_train_loss, train_acc, avg_val_loss, val_acc
	
	
def wandb_run(train_set, val_set, test_set):
	run = wandb.init()

	batch_size = wandb.config.batch_size
	lr = 10**wandb.config.log_lr
	kern_size = wandb.config.kern_size
	dropout = wandb.config.dropout


	train_loader = DataLoader(dataset = train_set, batch_size=batch_size, shuffle = True)
	val_loader = DataLoader(dataset = val_set, batch_size=batch_size, shuffle = True)
	# test_loader = DataLoader(dataset = test_set, batch_size=batch_size, shuffle = True)

	model = ConvNeuralNet(256, kern_size, dropout).to(device)

	optimizer = Adam(model.parameters(), lr=lr)
	criterion = MSELoss(reduction='sum')

	for epoch in range(50):
		model, train_loss, train_acc, val_loss, val_acc = train_one_epoch(model, train_loader, val_loader, optimizer, criterion)
		wandb.log({
			'epoch': epoch,
			'train_loss': train_loss, 
			'train_acc': train_acc,
			'val_loss': val_loss,
			'val_acc': val_acc
		})

def hyperparameter_sweep(train_set, val_set, test_set):
	sweep_configuration = {
		'method': 'random',
		'name': 'sweep',
		'metric': {'goal': 'minimize', 'name': 'train_loss'},
		'parameters': 
		{
			'batch_size': {'values': [4, 16, 64]},
			# 'epochs': {'values': [5, 10, 15]},
			'log_lr': {'values': [-6 + 0.5*i for i in range(9)]}, # 'max': -2.0, 'min': -7.0},
			'kern_size': {'max': 9, 'min': 3},
			'dropout': {'values': [0.25 + 0.05 * i for i in range(6)]}  # 'max': 0.5, 'min': 0.3}
		}
	}

	sweep_id = wandb.sweep(sweep=sweep_configuration, project='esinn_6conv4lin_nrows1000_epochs50')

	wandb.agent(sweep_id, function=lambda : wandb_run(train_set, val_set, test_set), count=100)

	return
