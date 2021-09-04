import json
import math
import numpy as np
import plotly.graph_objects as go
import plotly as py
import chart_studio
import plotly.io as io
import plotly.io as pio
pyplt = py.offline.plot
p=io.renderers['png']
p.width=800
p.height=600
import openpyxl
from plotly.graph_objs import *
from plotly.graph_objs.layout import *
chart_studio.tools.set_config_file(world_readable=True,
                             sharing='public')
pio.templates.default = "none"
def func1(x,y):
    # [e ^ (-1 / x) + e ^ (-1 / y)]
    return np.exp(-1/x)+np.exp(-1/y)
def func2(x,y):
    #[e^(-1/x-1/y)]
    return np.exp(-1/x-1/y)
def func3(x,y):
    return np.exp(-1/x)-np.exp(-1/y)
if __name__=="__main__":
    x=np.arange(-5,5,0.1)
    y=np.arange(-5,5,0.1)
    xx,yy=np.meshgrid(x,y)
    zz1=func1(xx.ravel(),yy.ravel())
    zz2=func2(xx.ravel(),yy.ravel())
    zz3=func3(xx.ravel(),yy.ravel())
    print(xx.ravel().shape,yy.ravel().shape,zz1.shape)
    figure1=go.Mesh3d(x=xx.ravel(),y=yy.ravel(),z=zz1,opacity=0.5,
                    colorscale=[[0, 'yellow'], [1.0, 'rgb(255, 20, 60)']])
    figure2=go.Mesh3d(x=xx.ravel(),y=yy.ravel(),z=zz2,opacity=0.5,
                    colorscale=[[0, 'green'],  [1.0, 'rgb(0, 255,0)']])
    figure3=go.Mesh3d(x=xx.ravel(),y=yy.ravel(),z=zz3,opacity=0.5,
                     colorscale=[[0, 'rgb(0, 255, 255)'], [1.0, 'rgb(30, 144, 255)']])
    data=[figure1,figure2,figure3]
    layout= dict(scene = dict(
        xaxis = dict(nticks=1, range=[-5,5],),
                     yaxis = dict(nticks=1, range=[-5,5],),
                     zaxis = dict(nticks=2, range=[-10,10],),),
    width=700,
    margin=dict(r=20, l=10, b=10, t=10))
    fig=Figure(layout=layout,data=data)
    fig.show()