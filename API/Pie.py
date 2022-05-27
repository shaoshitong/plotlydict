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


class Pie(object):
    def __init__(self, labels, values):
        """
        :param labels: ["name1","name2","name3"]
        :param values: [value1,value2,value3]
        Note:x=[["a","b","c"]] y=[[1,2,3]]
        """
        self.x_list = labels
        self.y_list = values
        if not isinstance(self.x_list[0], list) and not isinstance(self.x_list[0], np.ndarray):
            self.x_list = [self.x_list]
        if not isinstance(self.y_list[0], list) and not isinstance(self.y_list[0], np.ndarray):
            self.y_list = [self.y_list]
        self.name = "Pie"
        self.title = "Pie"
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
        self.hole=0

    def set_params(self, name, value):
        self.__setattr__(name, value)
    def get_object_name(self):
        return "pie"
    def set_layout(self):
        self.layout = dict(
            uniformtext_minsize=12,
            uniformtext_mode='hide',
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

    def show(self,colors=None):
            """
            color use to indicate the every pie color
            """
            self.data = []
            iter = 0
            for x, y in zip(self.x_list, self.y_list):
                if colors != None:
                    color = colors
                else:
                    color = init.get_color_list(y, self.r, self.g, self.b)
                base_data=go.Pie(labels=x,values=y,marker=dict(colors=color,
                                                               line=dict(color='#000000', width=2)),
                                 textposition='inside',hole=self.hole, pull=init.get_size_list(y,0,0.25),
                                 hoverinfo='label+percent',
                                 )
                self.data.append(base_data)
                iter += 1
            self.set_layout()
            fig = Figure(data=self.data, layout=self.layout)
            fig.show()
    def get_fig(self):
        """
        color use to indicate the every pie color
        """
        colors = None
        if hasattr(self, "colors"):
            colors = self.colors
        x=self.x_list[0]
        y=self.y_list[0]
        if colors != None:
            color = colors
        else:
            color = init.get_color_list(y, self.r, self.g, self.b)
        base_data = go.Pie(labels=x, values=y, marker=dict(colors=color,
                                                           line=dict(color='#000000', width=2)),
                           textposition='inside', hole=self.hole, pull=init.get_size_list(y, 0, 0.25),
                           hoverinfo='label+percent',
                           )
        return base_data
# x=["a","b","c","d"]
# y=[100,102,400,800]
# Pie([x],[y]).show()
