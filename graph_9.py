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

ORIGIN=np.array([[62.306,69.547,72.528,74.676],
   [0.15777621282479,0.130132411198919,0.080663911798687,0.066699991670833]])

AUGMENT=np.array([
    [66.917,72.504,74.765,76.163],
    [0.126143657082799,0.396378046258311,0.108551063252893,0.252852614074774]])

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
ax.set_ylim(60,76.5)
plt.xticks(x,x_name,rotation=60)
ax.set_xlabel("Few-Shot Ratio",font1)
plt.ylabel('Top-1 Test Accuracy [%]',font1)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)
plt.title("CIFAR-100")
plt.savefig("./graph9.png")
