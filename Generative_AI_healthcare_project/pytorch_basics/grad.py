import torch

x = torch.tensor(2,requires_grad=True,dtype=torch.float)
y = x*x
z = y*y*y
z.backward()
print(z.grad)