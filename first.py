import numpy as np
import plotly as py
import chart_studio
import plotly.io as pio
import plotly.graph_objects as go
from plotly.graph_objs import *
import torch
import math
import pandas as pd
import math
def b_expression(x):
    return 1/(1-(3/4)*(2/3)**(x-1))-2
def a(x):
    return (x-2)/(x+4)
if __name__=="__main__":
    pio.templates.default = "simple_white"
    pyplt = py.offline.plot
    p = pio.renderers['png']
    p.width = 800
    p.height = 600
    chart_studio.tools.set_config_file(world_readable=True, sharing='public')
    x=np.linspace(1,100,100)
    li=[]
    a_1=2
    li.append(a_1)
    for i in range(99):
        a_1=a(a_1)
        li.append(a_1)
    lj=[]
    for i in range(100):
        lj.append(b_expression(i+1)+0.5)

    figure1= go.Scatter(x=x,y=lj,opacity=0.5,name="func",mode='lines',line=dict(
                             width=2,
                             color="red"
                         ))

    figure2= go.Scatter(x=x,y=li,opacity=0.5,name="func",mode='lines',line=dict(
                             width=2,
                             color="blue"
                         ))
    data=[figure1,figure2]
    layout = dict(
        title="func",
        font=dict(
            family="Courier New, monospace",
            size=35,
            color="black",
        ),
        xaxis_title="x",
        yaxis_title="y",
        legend=dict(
            x=0.8,
            y=0.9,
            bgcolor="white",
            bordercolor="black",
            borderwidth=2)
    )
    fig = Figure(data=data,layout=layout)
    fig.show()
