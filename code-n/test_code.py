import numpy as np
from dataloading import ShapesDataset, ShapesDatasetLive
from argparser import parse_args
import pandas as pd
import torch
from torch.utils.data import DataLoader
from data_vis import show_image
from img_gen import Circle, Square, Rectangle, Ellipse, Triangle



data_name = 'Circ_BonW'
data_csv = '../%s/%s_labels.csv'%(data_name, data_name)

print('Reading database...')
df = pd.read_csv(data_csv)
print('Data read complete')
print(df)

hist = df.hist(column='shape_count')

