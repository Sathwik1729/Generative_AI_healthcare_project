import sklearn.metrics
import torch
import torch.nn as nn
import numpy as np
import sklearn
from sklearn import datasets
from sklearn.preprocessing import StandardScaler as sc
from sklearn.model_selection import train_test_split

x,y = datasets.load_breast_cancer(return_X_y=True)
x = torch.from_numpy(x.astype(np.float32))
y = torch.from_numpy(y.astype(np.float32)).unsqueeze(1)

x_train, x_test,y_train,  y_test = train_test_split(x,y, test_size=0.2, random_state=123)

#applying scale and transforming train, fitting test



SC = sc()
x_train = torch.from_numpy(SC.fit_transform(x_train)).float() #fit transform returnsnp array
x_test = torch.from_numpy(SC.transform(x_test)).float()




# print(x.shape)
# print(y.shape)

class LogisticReg(nn.Module):
    def __init__(self,n_input_features):
        super().__init__()
        self.layer = nn.Linear(n_input_features,1)

    def forward(self,x):
        
        return torch.sigmoid(self.layer(x))
#2) model and optimiser
model = LogisticReg(x.shape[1])

loss = nn.BCELoss()
lr = 0.01
optimiser = torch.optim.SGD(model.parameters(),lr)


num_epochs = 1000

#3)training 
for epoch in range(num_epochs):
    y_pred = model(x_train)
    l = loss(y_pred,y_train)
    l.backward()

    optimiser.step()
    optimiser.zero_grad()

    if((epoch+1)%10 == 0):
        print(l.item())

with torch.no_grad():
    y_pred = model.forward(x_test)
    print(sklearn.metrics.classification_report(y_test,y_pred.round()))