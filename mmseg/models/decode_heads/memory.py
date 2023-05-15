import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F



class FeatureMemory(nn.Module):
    def __init__(self) -> None:
        super(FeatureMemory,self).__init__()
        self.memory = nn.Parameter(torch.zeros([1,3,512,15,15]), requires_grad = False)

    def update(self,feats):
        return

    def forward(self,feats):
        return
