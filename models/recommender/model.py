import torch
import torch.nn as nn 
import torch.nn.functional as F 

from .constants import *

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        # TODO

        self.nonlinear = nn.ReLU()


    def forward(self, inputs):
        # TODO
        raise NotImplementedError