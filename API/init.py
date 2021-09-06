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
pio.templates.default = "simple_white"
pyplt = py.offline.plot
p = pio.renderers['png']
p.width = 800
p.height = 600
root=os.getcwd()
if root not in sys.path:
    sys.path.append(root)
root=os.path.dirname(os.path.realpath(__file__))
if root not in sys.path:
    sys.path.append(root)
chart_studio.tools.set_config_file(world_readable=True, sharing='public')
def turn_tensor_to_numpy(tensor):
    if torch.is_tensor(tensor):
        tensor=tensor.clone().detach().cpu().numpy()
        return tensor
    else:
        return tensor
def turn_numpy_to_tensor(tensor):
    if isinstance(tensor,np.ndarray):
        return torch.from_numpy(tensor)
    else:
        return tensor
def get_color(temp,r,g,b,_max,_min):
    temp=int(temp)
    # min_rgb=min(r,g,b)
    temp = 120 * (temp - _min) / (_max - _min + 1) - 1
    return 'rgb({r}, {g}, {b})'.format(
        r=(r-temp+255)%255,
        b=(b-temp+255)%255,
        g=g
    )
def get_color_list(temp_list,r,g,b):
    assert isinstance(temp_list,list) or isinstance(temp_list,np.ndarray)
    if isinstance(temp_list,np.ndarray):
        temp_list=temp_list.tolist()
    max_value=max(*temp_list)
    min_value=min(*temp_list)
    rgb_list=[]
    for i in temp_list:
        rgb_list.append(get_color(i,r,g,b,max_value,min_value))
    return rgb_list
def get_size(temp,old_min,old_max,new_min,new_max):
    temp=(new_max-new_min)*(temp-old_min)/(old_max-old_min+1)+new_min
    return float(temp)

def get_size_list(temp_list,new_min=12,new_max=24):
    assert isinstance(temp_list, list) or isinstance(temp_list, np.ndarray)
    if isinstance(temp_list, np.ndarray):
        temp_list = temp_list.tolist()
    max_value = max(*temp_list)
    min_value = min(*temp_list)
    size_list = []
    for i in temp_list:
        size_list.append(get_size(i, min_value,max_value,new_min,new_max))
    return size_list