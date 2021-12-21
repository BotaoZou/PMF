import math

import numpy as np
import torch
import yaml
from PIL import Image
from torch import nn
from torchvision.transforms import ToTensor

input = torch.tensor([[[1,1,1],[2,2,2]],[[1,1,1],[2,2,4]]], dtype=torch.float32)
print(input.mean(0))
BN = nn.BatchNorm1d(2)
output = BN(input)

print(output)
