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

def func(function,x,y):
    xx,yy=np.meshgrid(x,y)
    return function(xx,yy)

class Surface(object):
    def __init__(self, function,x,y):
        """
        :param function: like def function or lambda function
        :param x: 二元函数x轴方向坐标
        :param y: 二元函数y轴方向坐标
        Note:请在上述function,x,y外面加上[],该list表示其中存在多个Surface置入同一张图中
        """
        self.function_list=function
        self.x_list=x
        self.y_list=y
        if not isinstance(self.function_list, list):
            self.function_list = [self.function_list]
        if not isinstance(self.x_list[0], list) and not isinstance(self.x_list[0], np.ndarray):
            self.x_list = [self.x_list]
        if not isinstance(self.y_list[0], list) and not isinstance(self.y_list[0], np.ndarray):
            self.y_list = [self.y_list]
        self.name = "Surface"
        self.title = "Surface"
        self.family = "Courier New, monospace"
        self.font_size = 10
        self.font_color = "black"
        self.xaxis_title = ""
        self.yaxis_title = ""
        self.legend_x = 0.8
        self.legend_y = 0.9
        self.legend_bgcolor = "white"
        self.legend_bordercolor = "black"
        self.borderwidth = 2
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
        return "surface"

    def set_layout(self):
        self.layout = dict(
            title=self.title,
            font=dict(
                family=self.family,
                size=self.font_size,
                color=self.font_color,
            ),
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict( showgrid=False, zeroline=False),
            hovermode='closest',
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
        for x, y,function in zip(self.x_list, self.y_list,self.function_list):
            z=func(function,x,y)
            base_data=go.Surface(x=x,y=y,z=z,colorscale=colorscale,name=self.name,opacity=self.opacity)
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
        function =self.function_list[0]
        z = func(function, x, y)
        base_data = go.Surface(x=x, y=y, z=z, colorscale=colorscale, name=self.name, opacity=self.opacity)
        return base_data
# a=np.linspace(0,10,100)
# b=np.linspace(0,10,100)
# Surface([lambda x,y:np.log(np.exp(x)+np.exp(y))],[a],[b]).show()