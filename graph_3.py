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

CIFAR100_DATA=[[[70.38,0.333620828400832],
[74.09,0.124911397169582],
[72.75,0.191989583050749],
[75.35,0.296128993771939]]
,
[[70.49,0.516056475314753],
[74.72,0.13691498926602],
[73.27,0.221516418826062],
[76.18,0.215996031709583]]
,
[[70.76,0.280251587651082],
[74.87,0.244670818700983],
[74.04,0.082537148923263],
[76.86,0.158375683559242]]]


CIFAR100_DATA=np.array(CIFAR100_DATA)
CIFAR100_DATA=CIFAR100_DATA.transpose((1,0,2))

x=[r"$\times1$",r"$\times2$",r"$\times4$"]
fig = go.Figure(data=[
    go.Bar(name='vanilla CE w/o data augmentation', x=x,y=CIFAR100_DATA[0,:,0],error_y=dict(type="data",array=CIFAR100_DATA[0,:,1])),
    go.Bar(name='vanilla KD w/o data augmentation', x=x, y=CIFAR100_DATA[1,:,0],error_y=dict(type="data",array=CIFAR100_DATA[1,:,1])),
    go.Bar(name='vanilla CE with data augmentation', x=x, y=CIFAR100_DATA[2,:,0],error_y=dict(type="data",array=CIFAR100_DATA[2,:,1])),
    go.Bar(name='vanilla KD with data augmentation', x=x, y=CIFAR100_DATA[3,:,0],error_y=dict(type="data",array=CIFAR100_DATA[3,:,1])),

])
fig.add_shape(type="line",
    name='Teacher Model',
    x0=-0.5, y0=76.44, x1=2.5, y1=76.44,
    line=dict(
        color="rgb(0,0,0)",
        width=4,
        dash="dot",
    ),xref="x", yref="y",
)

# 柱状图模式需要设置：4选1

fig.update_layout(barmode='group')  # ['stack', 'group', 'overlay', 'relative']


layout = dict(
    title=r"$\textbf{CIFAR-100}$",
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
        range=[68,77.5],
    ),
    xaxis_title=r"$n_{step}$",
    yaxis_title=r"Top-1 Test Accuracy [%]",
    legend=dict(
        x=0.7,
        y=1.3,
        bgcolor="white",
        bordercolor="black",
        borderwidth=2)
)
fig.update_layout(layout)

fig.write_image("./graph3.png")

