# importing the libraries
import pandas as pd
import numpy as np
from argparser import parse_args
import pandas as pd

import matplotlib.pyplot as plt
import torch

from torch.utils.data import Dataset, DataLoader

# for evaluating the model
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from dataloading import ShapesDataset
from torch.optim import Adam, SGD
from torch.nn import Linear, ReLU, CrossEntropyLoss
from training import train,validateModel

def main(args):

	batch_size = 1000

	len_dataset = len(pd.read_csv(args.d))

	train_size = int(0.7 * len_dataset)
	val_size = int(0.1 * len_dataset)
	test_size = int(0.2 * len_dataset)

	dataset = ShapesDataset(csv_file = args.d)

	train_val_set,test_set =  torch.utils.data.random_split(dataset,[train_size+val_size,test_size])
	train_set,val_set =  torch.utils.data.random_split(train_val_set,[train_size,val_size])

	train_loader = DataLoader(dataset = train_set,batch_size=batch_size,shuffle = True)
	val_loader = DataLoader(dataset = val_set,batch_size=batch_size,shuffle = True)
	test_loader = DataLoader(dataset = test_set,batch_size=batch_size,shuffle = True)


	# checking if GPU is available
	if torch.cuda.is_available():
	    model = model.cuda()
	    criterion = criterion.cuda()

	model = train(train_loader,val_loader)

	validateModel(model,test_loader)



"""	# plotting the training and validation loss
	plt.plot(train_losses, label='Training loss')
	plt.plot(val_losses, label='Validation loss')
	plt.legend()
	plt.show()

	# prediction for training set
	with torch.no_grad():
		output = model(train_x.cuda())

	softmax = torch.exp(output).cpu()
	prob = list(softmax.numpy())
	predictions = np.argmax(prob, axis=1)

	# accuracy on training set
	accuracy_score(train_y, predictions)"""

if __name__ == '__main__':
	args = parse_args()

	main(args)