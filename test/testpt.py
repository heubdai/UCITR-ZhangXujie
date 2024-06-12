import torch
import torch.nn as nn

y = torch.load('../tempweights/NATOPS20.pt',map_location='cuda:0')
print(y)