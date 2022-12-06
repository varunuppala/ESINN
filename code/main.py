# importing the libraries
import pandas as pd
import numpy as np
from argparser import parse_args
# import pandas as pd

import matplotlib.pyplot as plt
import torch

from torch.utils.data import Dataset, DataLoader

# for evaluating the model
from sklearn.metrics import accuracy_score
# from tqdm import tqdm

from dataloading import ShapesDataset
from torch.optim import Adam, SGD
from torch.nn import Linear, ReLU, CrossEntropyLoss
from training import hyperparameter_sweep, train, wandb_run #, validateModel
from data_vis import dataset_iterate

import warnings
warnings.filterwarnings("ignore")

device = ("cuda" if torch.cuda.is_available() else "cpu")

def main(args):

	torch.manual_seed(42)
	np.random.seed(42)

	batch_size = 16						# Batch size
	nrows = None						# Number of datapoints to read from the csv (None = all of the data)
	dataset_name = 'Circ_WonB'			# Dataset to use
	num_models = 10						# How many models to train in this training run
	model_save_freq = 2					# After how many batches do we save a version of the model
	report_freq = 10						# After how many batches do we report the loss of the model


	csv_path = '../%s/%s_labels.csv'%(dataset_name, dataset_name)
	len_dataset = len(pd.read_csv(csv_path, nrows=nrows))
	train_size = int(0.7 * len_dataset)
	val_size = int(0.1 * len_dataset)
	test_size = int(0.2 * len_dataset)
	# test_size = len_dataset - train_size - val_size

	print('lengths of \n Dataset: {}, Train: {}, Validation: {} ,Test: {}'.format(len_dataset,train_size,val_size,test_size))

	dataset = ShapesDataset(csv_file = '../%s/%s_labels.csv'%(dataset_name, dataset_name), nrows=nrows ,device=device)

	train_val_set,test_set =  torch.utils.data.random_split(dataset,[train_size+val_size,test_size])
	train_set,val_set =  torch.utils.data.random_split(train_val_set,[train_size,val_size])

	train_loader = DataLoader(dataset = train_set,batch_size=batch_size,shuffle = True)
	val_loader = DataLoader(dataset = val_set,batch_size=batch_size,shuffle = True)
	test_loader = DataLoader(dataset = test_set,batch_size=batch_size,shuffle = True)

	print('Data loaded.')
	
	model_num_zeros = int(np.floor(np.log10(num_models)+1))

	for model_num in range(num_models):
		model_num_string = ('0'*model_num_zeros + str(model_num+1))[-model_num_zeros:]
		model_name = '%s_model%s'%(dataset_name, model_num_string)
		model_path = 'models/%s'%(model_name)
		loss_log_file = '%s/%s_loss_data.csv'%(model_path, model_name)
		logging_params = (model_name, model_path, model_save_freq, loss_log_file)
		
		model = train(logging_params=logging_params, report_freq=report_freq, cont_data_gen=False, dataset_params=None, train_loader=train_loader, val_loader=val_loader)

	# hyperparameter_sweep() # train_set, val_set, test_set)
	# wandb_run()

	# dataset_iterate(train_loader)

	# validateModel(model,test_loader)


if __name__ == '__main__':
	args = parse_args()

	main(args)