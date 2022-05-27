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
    return function(xx.ravel(),yy.ravel()),xx.ravel(),yy.ravel()

class Mash3D(object):
    def __init__(self, function,x,y,z=None):
        """
        :param function: like def function or lambda function
        :param x: 二元函数x轴方向坐标
        :param y: 二元函数y轴方向坐标
        :param z: 二元函数z的值
        Note:请在上述function,x,y外面加上[],该list表示其中存在多个Surface置入同一张图中，其中z和function只要存在一个即可
        """
        self.function_list=function
        self.x_list=x
        self.y_list=y
        if not isinstance(self.x_list[0], list) and not isinstance(self.x_list[0], np.ndarray):
            self.x_list = [self.x_list]
        if z !=None:
            self.z_list=z
        else:
            self.z_list=[["None"]*len(self.x_list)]
        if not isinstance(self.function_list, list):
            self.function_list = [self.function_list]
        if not isinstance(self.y_list[0], list) and not isinstance(self.y_list[0], np.ndarray):
            self.y_list = [self.y_list]
        if not isinstance(self.z_list[0], list) and not isinstance(self.z_list[0], np.ndarray):
            self.y_list = [self.z_list]
        self.name = "Mash3d"
        self.title = "Mash3d"
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
        self.all_mx=0.
        self.all_nx=0.
        for x in self.x_list:
            if isinstance(x,np.ndarray):
                m=np.max(x)
                n=np.min(x)
            elif isinstance(x,list):
                m=max(x)
                n=min(x)
            if m>self.all_mx:
                self.all_mx=m
            elif n<self.all_nx:
                self.all_nx=n
        self.all_my=0.
        self.all_ny=0.
        for y in self.y_list:
            if isinstance(y,np.ndarray):
                m=np.max(y)
                n=np.min(y)
            elif isinstance(y,list):
                m=max(y)
                n=min(y)
            if m>self.all_my:
                self.all_my=m
            elif n<self.all_ny:
                self.all_ny=n


    def set_params(self, name, value):
        self.__setattr__(name, value)

    def get_object_name(self):
        return "mesh3d"

    def set_layout(self):
        self.layout = dict(
            title=self.title,
            font=dict(
                family=self.family,
                size=self.font_size,
                color=self.font_color,
            ),
            xaxis = dict(nticks=5, range=[self.all_nx,self.all_mx],),
            yaxis = dict(nticks=5, range=[self.all_ny,self.all_my],),
            margin=dict(r=20, l=10, b=10, t=10),
            autosize=False,
            height=550,
            width=550,
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
        for x, y,function,z in zip(self.x_list, self.y_list,self.function_list,self.z_list):
            if z[0]=="None":
                z, x, y = func(function, x, y)
                base_data = go.Mesh3d(x=x, y=y, z=z, colorscale=colorscale, name=self.name, opacity=self.opacity)
                self.data.append(base_data)
            else:
                base_data = go.Mesh3d(x=x, y=y, z=z, colorscale=colorscale, name=self.name, opacity=self.opacity)
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
        z=self.z_list[0]
        if z[0] == "None":
            z, x, y = func(function, x, y)
            base_data = go.Mesh3d(x=x, y=y, z=z, colorscale=colorscale, name=self.name, opacity=self.opacity)
        else:
            base_data = go.Mesh3d(x=x, y=y, z=z, colorscale=colorscale, name=self.name, opacity=self.opacity)
        return base_data
# a=np.random.rand(100)
# b=np.random.rand(100)
# z=np.random.rand(100)
# Mash3D([lambda x,y:np.log(np.exp(x)+np.exp(y))],[a],[b],[z]).show()