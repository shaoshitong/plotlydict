import torch
import torch.nn as nn
import torch.nn.functional as F

A=torch.randn(1000,20)*0.01+0.01

B=torch.randn(1000,20)*0.02-0.2

C=torch.randn(1000,20)*2 + 2

D=torch.cat([A,B,C],0)

from TSNE import T_SNE

T_SNE(D,torch.arange(3).unsqueeze(-1).repeat(1,1000).view(-1))