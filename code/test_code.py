import numpy as np
from dataloading import ShapesDataset, ShapesDatasetLive
from argparser import parse_args
import pandas as pd
import torch
from torch.utils.data import DataLoader
from data_vis import show_image
from img_gen import Circle, Square, Rectangle, Ellipse, Triangle



args = parse_args()

batch_size = 16
nrows = 100

len_dataset = len(pd.read_csv(args.d, nrows=nrows))

train_size = int(0.7 * len_dataset)
val_size = int(0.1 * len_dataset)
test_size = int(0.2 * len_dataset)
# test_size = len_dataset - train_size - val_size

print('lengths of \n Dataset: {}, Train: {}, Validation: {} ,Test: {}'.format(len_dataset,train_size,val_size,test_size))

print('Creating dataset...')

dataset = ShapesDataset(csv_file = args.d, nrows=nrows)

print('Splitting dataset...')

train_val_set,test_set =  torch.utils.data.random_split(dataset,[train_size+val_size,test_size])
train_set,val_set =  torch.utils.data.random_split(train_val_set,[train_size,val_size])

print('Creating dataloaders...')

train_loader = DataLoader(dataset = train_set,batch_size=batch_size,shuffle = True)
val_loader = DataLoader(dataset = val_set,batch_size=batch_size,shuffle = True)
test_loader = DataLoader(dataset = test_set,batch_size=batch_size,shuffle = True)

print('Iterating through train_loader...')

p = 0
for images, labels in val_loader:
    print(images.shape)
    print(labels.shape)
    # p += 1
    # for i, img in enumerate(images):
    #     show_image(img)
    #     print(labels[i].item())
print('P:', p)

#######################################

print('NEW WAY!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

np.random.seed(0)
torch.manual_seed(0)

n = 20
img_dim = 256
minw, maxw = 8, 100
minh, maxh = 8, 100
mincount, maxcount = 1, 10
minpr, maxpr = 0.02, 0.25

shape_color = 1
bg_color = 0

shape_set = [Circle, Square] #, Triangle, Ellipse, Rectangle]
params = (n, img_dim, shape_set, shape_color, bg_color, minw, maxw, minh, maxh, mincount, maxcount, minpr, maxpr)

dataset = ShapesDatasetLive(params)

train_loader = DataLoader(dataset = dataset,batch_size=5,shuffle = True)

p = 0
for images, labels in train_loader:
    print(images.shape)
    print(labels.shape)
    # p += 1
    # for i, img in enumerate(images):
    #     show_image(img)
    #     print(labels[i].item())


