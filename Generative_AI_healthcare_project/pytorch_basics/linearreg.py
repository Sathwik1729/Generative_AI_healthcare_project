import numpy as np
import torch.nn as nn
import torch
from sklearn import datasets

x,y = datasets.make_regression(100,2,bias=1,random_state=2,noise=20)

x = torch.from_numpy(x.astype(np.float32))

y = torch.from_numpy(y.astype(np.float32)).unsqueeze(1)



model = nn.Linear(x.shape[1],y.shape[1])
optimiser = torch.optim.SGD(model.parameters(), lr = 0.01)
loss = nn.MSELoss()

n_iter = 200

print(f"prediction before: {(model(x) - y).mean()}")

for epoch in range(n_iter):
    model.zero_grad()
    y_pred = model(x)

    l = loss(y_pred,y)
    l.backward()
    optimiser.step()
    if((epoch+1)%10 == 0):
        print(f"Present loss is {l.item()}, epoch:{epoch}")

print(f"Final prediction : {(model(x) - y).mean()}")
# print(model.parameters())





















    