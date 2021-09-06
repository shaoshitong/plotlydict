import sys,os
def add_init():
    root=os.path.dirname(os.path.realpath(__file__))
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
class Bar(object):
    def __init__(self,x,y):
        self.x_list=x
        self.y_list=y
        if not isinstance(self.x_list[0],list) and not isinstance(self.x_list[0],np.ndarray):
            self.x_list=[self.x_list]
        if not isinstance(self.y_list[0],list) and not isinstance(self.y_list[0],np.ndarray):
            self.y_list=[self.y_list]
        self.name="Bar"
        self.title="Bar"
        self.family="Courier New, monospace"
        self.font_size=35
        self.font_color="black"
        self.xaxis_title=""
        self.yaxis_title=""
        self.legend_x=0.8
        self.legend_y=0.9
        self.legend_bgcolor="white"
        self.legend_bordercolor="black"
        self.borderwidth=2
        self.r=random.randint(0,256)
        self.g=random.randint(0,256)
        self.b=random.randint(0,256)
    def get_object_name(self):
        return "bar"
    def set_layout(self):
        self.layout = dict(
            title=self.title,
            font=dict(
                family=self.family,
                size=self.font_size,
                color=self.font_color,
            ),
            xaxis=dict(zeroline=False),
            yaxis=dict(zeroline=False),
            xaxis_title=self.xaxis_title,
            yaxis_title=self.yaxis_title,
            legend=dict(
                x=self.legend_x,
                y=self.legend_y,
                bgcolor=self.legend_bgcolor,
                bordercolor=self.legend_bordercolor,
                borderwidth=self.borderwidth)
        )

    def set_params(self,name,value):
        self.__setattr__(name,value)
    def show(self):
        self.data=[]
        iter=0
        for x,y in zip(self.x_list,self.y_list):
            base_data=go.Bar(x=x,y=y,name=self.name+str(iter),marker=
            dict(color=init.get_color_list(y,self.r,self.g,self.b)),)
            self.data.append(base_data)
            iter+=1
        self.set_layout()
        fig = Figure(data=self.data, layout=self.layout)
        fig.show()
    def get_fig(self):
        x=self.x_list[0]
        y=self.y_list[0]
        base_data = go.Bar(x=x, y=y, name=self.name + str(iter), marker=
        dict(color=init.get_color_list(y, self.r, self.g, self.b)))
        return base_data


# import numpy as np
# x=np.linspace(0,100,100)
# y=np.random.randint(1,10,(100))
# Bar([x],[y]).show()



