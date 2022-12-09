from pandas.core.common import flatten
import copy
import numpy as np
import random
import pandas as pd
from PIL import Image

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import Dataset, DataLoader

# for creating validation set
from sklearn.model_selection import train_test_split
# import albumentations as A
# from albumentations.pytorch import ToTensorV2
# import cv2

import img_gen

transform = transforms.Compose([
    transforms.ToTensor(),
                                     ])

class ShapesDataset(Dataset):
    def __init__(self, csv_file, transform=transform, nrows=None, device='cpu'):
        super().__init__()
        self.annotations = pd.read_csv(csv_file, nrows=nrows)
        self.transform = transform
        self.images = []
        self.labels = []
        for idx in range(len(self.annotations)):
            image_filepath = '../data/'+self.annotations.iloc[idx,0][1:]+'.png'
            image = Image.open(image_filepath).convert("L")
            label = torch.tensor(int(self.annotations.iloc[idx,1])).type(torch.FloatTensor)
            if self.transform is not None:
                image = self.transform(image)
            self.images.append(image)
            self.labels.append(label)

        
    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        # # print('Getting item', idx)
        # image_filepath = '.'+self.annotations.iloc[idx,0]+'.png'

        # image = Image.open(image_filepath).convert("L")

        # label = torch.tensor(int(self.annotations.iloc[idx,1])).type(torch.FloatTensor)

        # if self.transform is not None:
        #     image = self.transform(image)
        
        # return image, label
        return self.images[idx], self.labels[idx]


class ShapesDatasetLive(Dataset):
    def __init__(self, dataset_params):
        super().__init__()

        self.length = dataset_params[0]
        self.images, self.labels = img_gen.gen_epoch(dataset_params)
        self.images = torch.from_numpy(self.images).type(torch.FloatTensor)
        self.labels = torch.from_numpy(self.labels).type(torch.FloatTensor)

        
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # # print('Getting item', idx)
        # image_filepath = '.'+self.annotations.iloc[idx,0]+'.png'

        # image = Image.open(image_filepath).convert("L")

        # label = torch.tensor(int(self.annotations.iloc[idx,1])).type(torch.FloatTensor)

        # if self.transform is not None:
        #     image = self.transform(image)
        
        # return image, label
        return self.images[idx], self.labels[idx]
