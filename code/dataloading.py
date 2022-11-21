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

transform = transforms.Compose([
    transforms.ToTensor(),
                                     ])

class ShapesDataset(Dataset):
    def __init__(self, csv_file,transform=transform):
        super().__init__()
        self.annotations = pd.read_csv(csv_file, nrows=500)
        self.transform = transform
        
    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        image_filepath = '.'+self.annotations.iloc[idx,0]+'.png'

        image = Image.open(image_filepath).convert("L")

        label = torch.tensor(int(self.annotations.iloc[idx,1]))

        if self.transform is not None:
            image = self.transform(image)
        
        return image, label
