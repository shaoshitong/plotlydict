import API.api as api

import numpy as np
import os,sys
import plotly as py
import chart_studio
import plotly.io as pio
import plotly.graph_objects as go
from plotly.graph_objs import *
import matplotlib.pyplot as plt
import torch
import math
import pandas as pd
import math

CIFAR10_DATA=[[[91.57,0.232559260240931],
[92.75,0.238281546155166],
[91.00,0.23357062763806],
[92.89,0.167095638425882]]
,
[[92.25,0.114009189270408],
[93.24,0.212495378100995],
[91.79,0.13876324475609],
[93.28,0.244086245491417]]
,
[[92.72,0.114371991243276],
[93.47,0.181588231210852],
[92.34,0.24439039806561],
[93.51,0.142468041529119]]]
CIFAR10_DATA=np.array(CIFAR10_DATA)
CIFAR10_DATA=CIFAR10_DATA.transpose((1,0,2))

x=[r"$\times1$",r"$\times2$",r"$\times4$"]
fig = go.Figure(data=[
    go.Bar(name='vanilla CE w/o data augmentation', x=x,y=CIFAR10_DATA[0,:,0],error_y=dict(type="data",array=CIFAR10_DATA[0,:,1])),
    go.Bar(name='vanilla KD w/o data augmentation', x=x, y=CIFAR10_DATA[1,:,0],error_y=dict(type="data",array=CIFAR10_DATA[1,:,1])),
    go.Bar(name='vanilla CE with data augmentation', x=x, y=CIFAR10_DATA[2,:,0],error_y=dict(type="data",array=CIFAR10_DATA[2,:,1])),
    go.Bar(name='vanilla KD with data augmentation', x=x, y=CIFAR10_DATA[3,:,0],error_y=dict(type="data",array=CIFAR10_DATA[3,:,1])),

])

# 柱状图模式需要设置：4选1

fig.update_layout(barmode='group')  # ['stack', 'group', 'overlay', 'relative']


layout = dict(
    title=r"$\textbf{CIFAR-10}$",
    font=dict(
        family="Times New Roman",
        size=15,
        color="black",
    ),
    xaxis=dict(
        showgrid=False,
        zeroline=False,
        showline=True,
        showticklabels=True,
    ),
    yaxis=dict(
        showgrid=False,
        zeroline=False,
        showline=True,
        showticklabels=True,
        range=[90,95],
    ),
    xaxis_title=r"$n_{step}$",
    yaxis_title=r"Top-1 Test Accuracy [%]",
    legend=dict(
        x=0.7,
        y=1.1,
        bgcolor="white",
        bordercolor="black",
        borderwidth=2)
)
fig.update_layout(layout)
fig.write_image("./graph2.png")

