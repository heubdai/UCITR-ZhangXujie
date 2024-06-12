import torch
import numpy as np
import torch.nn.functional as F

a = np.array([[[1.0,2.0,3],[4,5,6]],[[7,8,9],[10,11,12]]])
b = F.max_pool1d(a.transpose(1,2),kernel_size=a.size(1)).transepose(1,2)
print(b)