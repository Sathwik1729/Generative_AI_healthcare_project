import numpy as np 

a = np.array([[1, 2, 3], [4, 5, 6]])  # shape: (2, 3)
b = np.array([[1, 0, 1],[1,1,1],[1,1,1]])  

print(np.dot(a, b))  # → shape: (2,)
