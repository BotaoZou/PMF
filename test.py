import math

import numpy as np
import torch
import yaml
from PIL import Image
from torch import nn
from torchvision.transforms import ToTensor

loss = nn.CrossEntropyLoss()

# a = torch.Tensor(np.arange(6).reshape((2,3,1,1)))
# b = torch.LongTensor(np.array([3,3]).reshape((2,1,1)))
# print(a)
# print(b)
# print(loss(a,b))
a = torch.tensor(np.ones(12).reshape((3,4,1)))
b = a.expand((3,4,2))
print(b.      shape)
c = a -\
        b
print(c)