import torch
from torch.optim import Adam, SGD
from torch.nn import Linear, ReLU, CrossEntropyLoss,MSELoss
from model_regression import ConvNeuralNet

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

from torch import nn, cuda, optim
import torch.nn.functional as F


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
	return torch.mean((torch.abs(output - labels) <= 0.5).type(torch.FloatTensor)).item()


def train(train_loader,val_loader):

	counter = 0

	# number of epochs
	num_epochs = 100

	#Change classes to 1 for regression
	model = ConvNeuralNet()
	if device == "cuda":
		model = model.cuda()
	print(model)

	# defining the optimizer
	optimizer = Adam(model.parameters(), lr=1e-5)

	# defining the loss function 
	#Change this for regression
	#criterion = MSELoss()
	criterion = MSELoss()
	
	valAccHist = []
	trainLossHist = []
	valLossHist = []
	
	# We use the pre-defined number of epochs to determine how many iterations to train the network on
	for epoch in range(num_epochs):
		
		#Load in the data in batches using the train_loader object
		for i, (images, labels) in enumerate(train_loader):  
	        
	        # Move tensors to the configured device
			images = images.to(device)

			labels = labels.type(torch.FloatTensor)
			labels = labels.to(device)
	        #print(labels)

			model.train()
	        
	        # Forward pass
			outputs = model(images)
	        
			loss = criterion(outputs, labels)
	        
	        # Backward and optimize
			optimizer.zero_grad()

			# print('Loss:\n', loss)
			# print(loss.dtype)
	        
			loss.backward()
	        
			optimizer.step()

	        # Validate model every n iterations.
			if counter % 1 == 0:

				valLoss = 0
				valAcc = 0

	            # Validating the model
				model.eval()
	            
				with torch.no_grad():
					for inputs, labels in val_loader:

						inputs, labels = inputs.to(device), labels.to(device)
						output = model.forward(inputs)
						valLoss += criterion(output, labels).item()

						valAcc += accuracy(output, labels)
						# #Change these for regression too
						# output = torch.round(output)
						# top_p, top_class = output.topk(1, dim=1)
						# equals = top_class == labels.view(*top_class.shape)
						# valAcc += torch.mean(equals.type(torch.FloatTensor)).item()


	            # Output statistics.
				valAccHist += [valAcc / len(val_loader)]
				valLossHist += [valLoss / len(val_loader)]


		print('Epoch [{}/{}], TrainLoss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))
		print('\tVal Accuracy: {:.6f} \tTrain Loss: {:.6f} \tVal Loss: {:.6f}'.format( valAcc/len(val_loader), loss.item(), valLoss/len(val_loader)))

		counter+=1

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
