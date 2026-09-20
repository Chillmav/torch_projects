import torch
import numpy as np

data = [[1,2], [3,4]]

x_data = torch.tensor(data)
np_arr = np.array(data)
x_np = torch.from_numpy(np_arr)

shape = (2,3)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)
