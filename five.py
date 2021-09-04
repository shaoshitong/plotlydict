import json
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


def func1(x, y):
    # [e ^ (-1 / x) + e ^ (-1 / y)]
    return np.exp(-1 / x) + np.exp(-1 / y)


def func2(x, y):
    # [e^(-1/x-1/y)]
    return np.exp(-1 / x - 1 / y)


def func3(x, y):
    # e ^ (-1 / x) -e ^ (-1 / y)
    return np.exp(-1 / x) - np.exp(-1 / y)


if __name__ == "__main__":
    x = np.linspace(-20, 20, 100)
    y = np.linspace(-20, 20, 100)
    xx, yy = np.meshgrid(x, y)
    zz1 = func1(xx, yy)
    zz2 = func2(xx, yy)
    zz3 = func3(xx, yy)
    print(x, y, zz1)
    fig = make_subplots(
        rows=1, cols=3,
        specs=[[{'type': 'surface'}, {'type': 'surface'}, {'type': 'surface'}]]
        , subplot_titles=["e ^ (-1 / x) + e ^ (-1 / y)", "e^(-1 / x - 1 / y)", 'e ^ (-1 / x) -e ^ (-1 / y)'],
    )

    print(xx, yy, zz1)
    print(x)
    fig.add_trace(
        go.Surface(x=x, y=y, z=zz1, colorscale='Viridis', showscale=False, name="e ^ (-1 / x) + e ^ (-1 / y)"),
        row=1, col=1)
    fig.add_trace(
        go.Surface(x=x, y=y, z=zz2, colorscale='RdBu', showscale=False, name="e^(-1 / x - 1 / y)"),
        row=1, col=2)
    fig.add_trace(
        go.Surface(x=x, y=y, z=zz3, colorscale='YlOrRd', showscale=False, name='e ^ (-1 / x) -e ^ (-1 / y)'),
        row=1, col=3)
    fig.update_layout(
        title_text='different function of x and y',
        height=800,
        width=2000,
        legend=dict(
            x=0.1,
            y=0.9,
            bgcolor="white",
            bordercolor="black",
            borderwidth=2)
    )
    fig.show()
