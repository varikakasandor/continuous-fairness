import os
import sys
sys.path.append(os.path.abspath(os.path.join('../..')))
import torch
from torch import nn
import torch.nn.functional as F


class SimpleNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SimpleNN, self).__init__()
        size = 50
        self.first = nn.Linear(input_size, size)
        self.fc = nn.Linear(size, size)
        self.last = nn.Linear(size, num_classes)

    def forward(self, x):
        out = F.selu(self.first(x))
        out = F.selu(self.fc(out))
        out = self.last(out)
        out = torch.sigmoid(out)
        return out