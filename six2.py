import math
import numpy as np
import plotly.graph_objects as go
import plotly as py
import chart_studio
import plotly.io as io
import plotly.io as pio
from plotly.subplots import make_subplots

pyplt = py.offline.plot
p = io.renderers['png']
p.width = 800
p.height = 600
import openpyxl
from plotly.graph_objs import *
from plotly.graph_objs.layout import *

chart_studio.tools.set_config_file(world_readable=True,
                                   sharing='public')
pio.templates.default = "none"

def get_torch(data):
    if len(*data.shape)!=2:
        assert((len(*data.shape)!=2) and "the dim is not two")
    data=data.numpy()
    x=np.linspace(0,data.shape[0]-1,data.shape[0])
    y=np.linspace(0,data.shape[1]-1,data.shape[1])
    figure1=go.Surface(x=x,y=y,z=data,opacity = .7)
    data=[figure1]
    fig=Figure(data=data)
    camera = dict(
        up=dict(x=0, y=0, z=1),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=1.25, y=1.25, z=1.25)
    )
    name='torch_vis'
    fig.update_layout(name=name,scene_camera=camera)
    fig.show()

