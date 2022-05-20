import sys

import numpy as np

root='.'
sys.path.append(root)
import yaml
import copy
import torch
from torch.utils.data import DataLoader
import argparse
import os
import matplotlib.pyplot as plt
font1 = {'family':"Times New Roman",
         'weight': 'bold',
         'style': 'normal',
         'size': 30,
         }
plt.rc('font',**font1)
import seaborn as sns


fig, ax = plt.subplots(1, 1, figsize=(20,15), dpi=100)

ORIGIN=np.array([[87.838,90.865,92.342,93.104],
   [0.098183501669068,0.137457064003434,0.093428522887224,0.106374600142872]])

AUGMENT=np.array([
    [90.112,92.132,92.894,93.144],
    [0.188726727777964,0.092574294488269,0.184101421323501,0.16547574511759]])

x_name=[
    f"{i}%" for i in [25,50,75,100]
]
x=[0,1,2,3]
ax.plot(x, ORIGIN[0,:],linestyle='--',marker = "s",color = "tab:green",linewidth=6, markersize=2,label="vanilla KD w/o data augmentation")
ax.plot(x, AUGMENT[0,:], linestyle='--',  marker = "s",color = "tab:red",linewidth=6, markersize=2,label="vanilla KD with data augmentation")
ax.errorbar(x,ORIGIN[0,:],\
            yerr=ORIGIN[1,:],\
            fmt="s",color = "tab:green",ecolor='tab:green',elinewidth=4,capsize=20,ms=15,capthick=4,)
ax.errorbar(x,AUGMENT[0,:],\
            yerr=AUGMENT[1,:],\
            fmt="s",color = "tab:red",ecolor='tab:red',elinewidth=4,capsize=20,ms=15,capthick=4)
ax.legend()
depth=4
ax.axvspan(-0.5, 3.5, color="tab:gray", alpha=0.15, lw=0)
for pool in x:
    ax.axvspan(pool - 1.0 / depth, pool + 1.0 / depth, color="tab:blue", alpha=0.15, lw=0)

ax.set_xlim(-0.5,3.5)
ax.set_ylim(87,94)
plt.xticks(x,x_name,rotation=60)
ax.set_xlabel("Few-Shot Ratio",font1)
plt.ylabel('Top-1 Test Accuracy [%]',font1)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)
plt.title("CIFAR-10")
plt.savefig("./graph8.png")
