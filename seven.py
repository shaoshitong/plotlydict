
import pandas as pd
import plotly.graph_objects as go
import chart_studio
import numpy as np
import plotly as py
import chart_studio
import plotly.io as pio
from plotly.graph_objs import *
import torch
import math
# Read data from a csv
z_data = pd.read_csv('mt_bruno_elevation.csv')
print(z_data)
fig = go.Figure(data=go.Surface(z=z_data, showscale=False))
fig.update_layout(
    title='Mt Bruno Elevation',
    width=400, height=400,
    margin=dict(t=40, r=0, l=20, b=20)
)

name = 'default'
# Default parameters which are used when `layout.scene.camera` is not provided
camera = dict(
    up=dict(x=0, y=0, z=1),
    center=dict(x=0, y=0, z=0),
    eye=dict(x=1.25, y=1.25, z=1.25)
)

fig.update_layout(scene_camera=camera, title=name)
fig.show()