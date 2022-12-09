import os
import csv

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
from dataloading import ShapesDatasetLive
from img_gen import Circle, Ellipse, Square, Rectangle, Triangle


device = ("cuda" if torch.cuda.is_available() else "cpu")

#device = ("mps" if torch.backends.mps.is_built() else "cpu")

#ref : https://www.kaggle.com/code/liamvinson/transfer-learning-cnn

def accuracy(output, labels):
	return torch.sum((torch.abs(output - labels) <= 0.5).type(torch.FloatTensor)).item()


def train(logging_params=None, report_freq=100, cont_data_gen=True, dataset_params=None, train_loader=None, val_loader=None):
	assert cont_data_gen or ((train_loader is not None) and (val_loader is not None))	# Either use continuous data generation or input a train and val loader
	assert not cont_data_gen or dataset_params is not None								# Cannot turn on continuous data generation without setting the dataset parameters
	
	if logging_params is not None:
		model_name, model_path, model_save_freq, loss_log_file = logging_params
		# model_path = 'models/%s'%(model_name)
		if not os.path.exists(model_path):
			os.makedirs(model_path)
		with open(loss_log_file, 'w', newline='') as f:
			csvwriter = csv.writer(f)
			csvwriter.writerow(['epoch_num', 'batch_num', 'avg_train_loss', 'train_acc', 'avg_val_loss', 'val_acc'])

	num_epochs = 3
	if cont_data_gen:
		batch_size = 16
	lr = 10**-3.5

	model = ConvNeuralNet(256, k_size=None, dropout_val=0.0)
	if device == "cuda":
		model = model.cuda()
	print(model)

	if cont_data_gen:
		val_data = ShapesDatasetLive(dataset_params)
		val_loader = DataLoader(dataset = val_data, batch_size=batch_size, shuffle = True)

	optimizer = Adam(model.parameters(), lr=lr)
	criterion = nn.MSELoss(reduction='sum')
	
	# We use the pre-defined number of epochs to determine how many iterations to train the network on
	for epoch in range(num_epochs):

		if cont_data_gen:
			train_data = ShapesDatasetLive(dataset_params)
			train_loader = DataLoader(dataset = train_data, batch_size=batch_size, shuffle = True)

		report_params = (epoch, num_epochs, report_freq)

		model, avg_train_loss, train_acc, avg_val_loss, val_acc = train_one_epoch(model, train_loader, val_loader, optimizer, criterion, report_params, logging_params)

		print('='*10)
		print('Epoch [{}/{}] Complete'.format(epoch+1, num_epochs))
		print('\tTrain Loss: {:.4f}\tTrain Accuracy: {:.4f}\tVal Loss: {:.4f}\tVal Accuracy: {:.4f}'.format(avg_train_loss, train_acc, avg_val_loss, val_acc))
		print('='*10)

		torch.save(model, '%s/%s.pth'%(model_path, model_name))

	return model


# def validateModel(model, valLoader):
#     '''
#     Overview:
#         Predicts a validation set using the CNN model and prints accuracy values.
        
#     Inputs:
#         model - Machine learning model.
#         valLoader - Validation set.
#     '''
#     valLoss = 0
#     val1Acc = 0
#     val5Acc = 0
    
#     # Confusion matrix lists
#     pred = []
#     true = []
    
#     criterion = nn.NLLLoss()
    
#     print('Model Validation.')
    
#     model.eval()
#     with torch.no_grad():
#         for inputs, labels in valLoader:
            
#             # Calculates loss value.
#             inputs, labels = inputs.to(device), labels.to(device)
#             output = model.forward(inputs)
#             valLoss += criterion(output, labels).item()

#             val1Acc += accuracy(output, labels)
#             # # Calculates validation top 1 accuracy.
#             # output = torch.exp(output)
#             # _, top_class = output.topk(1, dim=1)
#             # equals = top_class == labels.view(*top_class.shape)
#             # val1Acc += torch.mean(equals.type(torch.FloatTensor)).item()
            
#             true += labels.tolist()
#             pred += top_class.flatten().tolist()
            
#             # Calculates validation top 5 accuracy.
#             _, top_class = output.topk(5, dim=1)
#             equals = top_class == labels.view(labels.shape[0], 1)
#             val5Acc += torch.sum(equals).item() / labels.shape[0]
            
#     # Calculates average.
#     n = len(valLoader)
#     valLoss /= n
#     val1Acc /= n
#     val5Acc /= n
    
#     print('Loss: {:.6f} \t Top-1 Accuracy: {:.6f} \t Top-5 Accuracy: {:.6f}\n'.format(valLoss, val1Acc, val5Acc))
    
#     # Confusion matrix
#     cm = confusion_matrix(true, pred)
#     plt.figure(figsize=(12, 9))
#     plt.title('Confusion Matrix')
#     sns.heatmap(cm)
#     plt.show()


def train_one_epoch(model, train_loader, val_loader, optimizer, criterion, report_params, logging_params):
	epoch, num_epochs, report_freq = report_params
	assert report_freq <= len(train_loader)	# Ensure that data is logged at least once

	if logging_params is not None:
		model_name, model_path, model_save_freq, loss_log_file = logging_params
		e_lead_zeros = int(np.floor(np.log10(num_epochs)+1))
		b_lead_zeros = int(np.floor(np.log10(len(train_loader))+1))

	model.train()
	
	train_loss = 0
	train_correct = 0
	train_pts = 0
	epoch_train_loss = 0
	epoch_train_correct = 0
	epoch_train_pts = 0

	#Load in the data in batches using the train_loader object
	for i, (images, labels) in enumerate(train_loader):  
		
		# Move tensors to the configured device
		images = images.to(device)
		labels = labels.to(device)
		
		# Forward pass
		outputs = model(images)
		outputs = torch.squeeze(outputs, 1)
		loss = criterion(outputs, labels)

		train_loss += loss.item()
		train_correct += accuracy(outputs, labels)
		train_pts += len(outputs)
		epoch_train_loss += loss.item()
		epoch_train_correct += accuracy(outputs, labels)
		epoch_train_pts += len(outputs)
		
		# Backward and optimize
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		if (i + 1) % report_freq == 0:
			avg_train_loss = train_loss / train_pts
			train_acc = train_correct / train_pts
			train_loss, train_correct, train_pts = 0, 0, 0

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

			print('Epoch [{}/{}], Batch [{:d}/{:d}]'.format(epoch+1, num_epochs, i+1, len(train_loader)))
			print('\tTrain Loss: {:.4f}\tTrain Accuracy: {:.4f}\tVal Loss: {:.4f}\tVal Accuracy: {:.4f}'.format(avg_train_loss, train_acc, avg_val_loss, val_acc))

			with open(loss_log_file, 'a', newline='') as f:
				csvwriter = csv.writer(f)
				csvwriter.writerow([epoch+1, i+1, avg_train_loss, train_acc, avg_val_loss, val_acc])

			model.train()
		
		if (i + 1) % model_save_freq == 0:
			e_num = ('0'*e_lead_zeros + str(epoch+1))[-e_lead_zeros:]
			b_num = ('0'*b_lead_zeros + str(i+1))[-b_lead_zeros:]
			torch.save(model, '%s/%s_e%s_b%s.pth'%(model_path, model_name, e_num, b_num))
	
	avg_train_loss = epoch_train_loss / epoch_train_pts
	train_acc = epoch_train_correct / epoch_train_pts

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
	
	
def wandb_run(): # train_set, val_set, test_set):
	run = wandb.init()

	batch_size = wandb.config.batch_size
	lr = 10**wandb.config.log_lr
	# kern_size = wandb.config.kern_size
	# dropout = wandb.config.dropout
	# batch_size = 16
	# lr = 1e-3
	# dropout = 0.3

	n = 1000
	img_dim = 256
	minw, maxw = 8, 100
	minh, maxh = 8, 100
	mincount, maxcount = 1, 10
	minpr, maxpr = 0.02, 0.25
	shape_color = 1
	bg_color = 0
	shape_set = [Circle] #, Square] # , Triangle, Ellipse, Rectangle]
	params = (n, img_dim, shape_set, shape_color, bg_color, minw, maxw, minh, maxh, mincount, maxcount, minpr, maxpr)

	val_data = ShapesDatasetLive(params)
	val_loader = DataLoader(dataset = val_data, batch_size=batch_size, shuffle = True)

	# train_loader = DataLoader(dataset = train_set, batch_size=batch_size, shuffle = True)
	# val_loader = DataLoader(dataset = val_set, batch_size=batch_size, shuffle = True)
	# test_loader = DataLoader(dataset = test_set, batch_size=batch_size, shuffle = True)

	model = ConvNeuralNet(256, None, 0).to(device)

	optimizer = Adam(model.parameters(), lr=lr)
	criterion = MSELoss(reduction='sum')

	for epoch in range(25):
		train_data = ShapesDatasetLive(params)
		train_loader = DataLoader(dataset = train_data, batch_size=batch_size, shuffle = True)

		# for images, labels in train_loader:
		# 	for image in images:
		# 		show_image(image)

		model, train_loss, train_acc, val_loss, val_acc = train_one_epoch(model, train_loader, val_loader, optimizer, criterion)
		wandb.log({
			'epoch': epoch,
			'train_loss': train_loss, 
			'train_acc': train_acc,
			'val_loss': val_loss,
			'val_acc': val_acc
		})
		# print('Complete epoch', epoch)
	# torch.save(model, 'models/%s.pth'%(run.name))

def hyperparameter_sweep(): #train_set, val_set, test_set):
	sweep_configuration = {
		'method': 'random',
		'name': 'sweep',
		'metric': {'goal': 'minimize', 'name': 'train_loss'},
		'parameters': 
		{
			'batch_size': {'values': [4, 8, 16, 32, 64]},
			# 'epochs': {'values': [5, 10, 15]},
			'log_lr': {'values': [i / 2 for i in range(-10, -3)]}, # [-5 + 0.5*i for i in range(9)]}, # 'max': -2.0, 'min': -7.0},
			# 'kern_size': {'max': 9, 'min': 3},
			# 'dropout': {'values': [0.3]} # [0.25 + 0.05 * i for i in range(6)]}  # 'max': 0.5, 'min': 0.3}
		}
	}

	sweep_id = wandb.sweep(sweep=sweep_configuration, project='esinn_livedata_circle_nodropout_esize1000')

	wandb.agent(sweep_id, function=lambda : wandb_run(), count=100)

	return
