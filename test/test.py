import numpy as np
import torch

v = torch.autograd.Variable(torch.FloatTensor([1]))
print v.data.numpy()
