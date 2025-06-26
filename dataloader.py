import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math

class WineData(Dataset):
    def __init__(self):
        data = np.loadtxt('./wine.csv',skiprows=1, dtype=np.float32,delimiter=',')
        self.x, self.y = data[:, 1:], data[:,0]
        # 
    
    def __getitem__(self, index):
        return self.x[index], self.y[index] 

    def __len__(self):
        return len(self.x)
    

data = WineData()

loader  = DataLoader(data, batch_size=math.ceil(len(data)/4), shuffle=True, num_workers=2)



for i,(features, labels) in enumerate(loader):
    print(features[0])
    print(labels[0])
    print(features.shape)
    print(labels.shape)