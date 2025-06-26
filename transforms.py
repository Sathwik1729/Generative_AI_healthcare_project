import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math

class WineData(Dataset):
    def __init__(self,transforms = None):
        self.transforms = transforms
        data = np.loadtxt('./wine.csv',skiprows=1, dtype=np.float32,delimiter=',')
        self.x, self.y = data[:, 1:], data[:,0]
        # 
    
    def __getitem__(self, index):
        sample = self.x[index], self.y[index]

        if self.transforms:
            sample = self.transforms(sample)

        return sample
    
    def __len__(self):
        return len(self.x)
    

data = WineData()

loader  = DataLoader(data, batch_size=math.ceil(len(data)/4), shuffle=True, num_workers=2)

class Mult:
    def __init__(self,factor):
        self.fac = factor

    def __call__(self, sample):
        return self.fac*sample
    
data = WineData()

loader  = DataLoader(data, batch_size=math.ceil(len(data)/4), shuffle=True, num_workers=2)


for i,(features, labels) in enumerate(loader):
    print(features[0])
    print(labels[0])
    print(features.shape)
    print(labels.shape)