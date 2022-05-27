import sys, os


def add_init():
    root = os.path.dirname(os.path.realpath(__file__))
    if root not in sys.path:
        sys.path.append(root)


add_init()
import init
import random
import plotly as py
import chart_studio
import plotly.io as pio
import plotly.graph_objects as go
from plotly.graph_objs import *
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


class Contour(object):
    def __init__(self, x, y, z):
        """
        :param x: x轴坐标
        :param y: y轴坐标
        :param z: 由 param x与 param y对应的值，
        Note:x=[[1,2,3]] y=[[1,2,3]] z=[[[2,3,4],[5,6,7],[8,9,10]]]
        """
        self.x_list = x
        self.y_list = y
        self.z_list = z

        if not isinstance(self.x_list[0], list) and not isinstance(self.x_list[0], np.ndarray):
            self.x_list = [self.x_list]
        if not isinstance(self.y_list[0], list) and not isinstance(self.y_list[0], np.ndarray):
            self.y_list = [self.y_list]
        if not isinstance(self.z_list[0], list) and not isinstance(self.z_list[0], np.ndarray):
            self.y_list = [self.z_list]
        self.name = "Contour"
        self.title = "Contour"
        self.family = "Courier New, monospace"
        self.font_size = 35
        self.font_color = "black"
        self.xaxis_title = ""
        self.yaxis_title = ""
        self.legend_x = 0.8
        self.legend_y = 0.9
        self.legend_bgcolor = "white"
        self.legend_bordercolor = "black"
        self.borderwidth = 2
        self.r = random.randint(0, 256)
        self.g = random.randint(0, 256)
        self.b = random.randint(0, 256)
        self.width = 4
        self.size = 8
        self.hoverinfo="skip"
        self.opacity=0.4
        self.colorscale_list = ['aggrnyl', 'agsunset', 'algae', 'amp', 'armyrose', 'balance', 'blackbody', 'bluered',
                                'blues', 'blugrn', 'bluyl', 'brbg', 'brwnyl', 'bugn', 'bupu', 'burg', 'burgyl',
                                'cividis', 'curl', 'darkmint', 'deep', 'delta', 'dense', 'earth', 'edge', 'electric',
                                'emrld', 'fall', 'geyser', 'gnbu', 'gray', 'greens', 'greys', 'haline', 'hot', 'hsv',
                                'ice', 'icefire', 'inferno', 'jet', 'magenta', 'magma', 'matter', 'mint', 'mrybm',
                                'mygbm', 'oranges', 'orrd', 'oryel', 'peach', 'phase', 'picnic', 'pinkyl', 'piyg',
                                'plasma', 'plotly3', 'portland', 'prgn', 'pubu', 'pubugn', 'puor', 'purd', 'purp',
                                'purples', 'purpor', 'rainbow', 'rdbu', 'rdgy', 'rdpu', 'rdylbu', 'rdylgn', 'redor',
                                'reds', 'solar', 'spectral', 'speed', 'sunset', 'sunsetdark', 'teal', 'tealgrn',
                                'tealrose', 'tempo', 'temps', 'thermal', 'tropic', 'turbid', 'twilight', 'viridis',
                                'ylgn', 'ylgnbu', 'ylorbr', 'ylorrd']


    def set_params(self, name, value):
        self.__setattr__(name, value)

    def get_object_name(self):
        return "contour"

    def set_layout(self):
        self.layout = dict(
            title=self.title,
            font=dict(
                family=self.family,
                size=self.font_size,
                color=self.font_color,
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
            ),
            xaxis_title=self.xaxis_title,
            yaxis_title=self.yaxis_title,
            legend=dict(
                x=self.legend_x,
                y=self.legend_y,
                bgcolor=self.legend_bgcolor,
                bordercolor=self.legend_bordercolor,
                borderwidth=self.borderwidth)
        )

    def show(self, colorscale=None):
        """
        colorscale could be
        ['aggrnyl', 'agsunset', 'algae', 'amp', 'armyrose', 
        'balance', 'blackbody', 'bluered', 'blues', 'blugrn',
        'bluyl', 'brbg', 'brwnyl', 'bugn', 'bupu', 'burg', 
        'burgyl', 'cividis', 'curl', 'darkmint', 'deep', 
        'delta', 'dense', 'earth', 'edge', 'electric',
        'emrld', 'fall', 'geyser', 'gnbu', 'gray', 'greens',
        'greys', 'haline', 'hot', 'hsv', 'ice', 'icefire', 
        'inferno', 'jet', 'magenta', 'magma', 'matter', 
        'mint', 'mrybm', 'mygbm', 'oranges', 'orrd', 'oryel',
        'peach', 'phase', 'picnic', 'pinkyl', 'piyg','plasma',
        'plotly3', 'portland', 'prgn', 'pubu', 'pubugn', 
        'puor', 'purd', 'purp', 'purples', 'purpor', 
        'rainbow', 'rdbu', 'rdgy', 'rdpu', 'rdylbu',
        'rdylgn', 'redor', 'reds', 'solar', 'spectral',
        'speed', 'sunset', 'sunsetdark', 'teal', 'tealgrn',
        'tealrose', 'tempo', 'temps', 'thermal', 'tropic',
        'turbid', 'twilight', 'viridis', 'ylgn', 'ylgnbu', 'ylorbr', 'ylorrd']
        """
        self.data = []
        self.random=random.randint(0,len(self.colorscale_list))
        if colorscale==None:
            colorscale=self.colorscale_list[self.random]
        iter = 0
        for x, y, z in zip(self.x_list, self.y_list, self.z_list):
            base_data = go.Contour(x=x,y=y,z=z,colorscale=colorscale,showscale=True,
                                   name=self.name, hoverinfo='skip',opacity=self.opacity)
            self.data.append(base_data)
            iter += 1
        self.set_layout()
        fig = Figure(data=self.data, layout=self.layout)
        fig.show()

    def get_fig(self):
        """
             colorscale could be
             ['aggrnyl', 'agsunset', 'algae', 'amp', 'armyrose', 
             'balance', 'blackbody', 'bluered', 'blues', 'blugrn',
             'bluyl', 'brbg', 'brwnyl', 'bugn', 'bupu', 'burg', 
             'burgyl', 'cividis', 'curl', 'darkmint', 'deep', 
             'delta', 'dense', 'earth', 'edge', 'electric',
             'emrld', 'fall', 'geyser', 'gnbu', 'gray', 'greens',
             'greys', 'haline', 'hot', 'hsv', 'ice', 'icefire', 
             'inferno', 'jet', 'magenta', 'magma', 'matter', 
             'mint', 'mrybm', 'mygbm', 'oranges', 'orrd', 'oryel',
             'peach', 'phase', 'picnic', 'pinkyl', 'piyg','plasma',
             'plotly3', 'portland', 'prgn', 'pubu', 'pubugn', 
             'puor', 'purd', 'purp', 'purples', 'purpor', 
             'rainbow', 'rdbu', 'rdgy', 'rdpu', 'rdylbu',
             'rdylgn', 'redor', 'reds', 'solar', 'spectral',
             'speed', 'sunset', 'sunsetdark', 'teal', 'tealgrn',
             'tealrose', 'tempo', 'temps', 'thermal', 'tropic',
             'turbid', 'twilight', 'viridis', 'ylgn', 'ylgnbu', 'ylorbr', 'ylorrd']
             """
        colorscale=None
        if hasattr(self, "colorscale"):
            colorscale = self.colorscale_list[random.randint(0,len(self.colorscale_list))]
        x = self.x_list[0]
        y = self.y_list[0]
        z = self.z_list[0]
        base_data = go.Contour(x=x,y=y,z=z,colorscale=colorscale,showscale=True,
                                   name=self.name, hoverinfo='skip',opacity=self.opacity)
        return base_data
# a=np.arange(0,100,1)
# b=np.arange(0,90,1)
# c=np.random.randn(100,90)
# iter=0.1
# for l in c:
#     l+=iter
#     iter+=0.1
# p=Contour([a],[b],[c])
# p.show()