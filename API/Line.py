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


class Line(object):
    def __init__(self, x, y):
        """
        :param x: x轴坐标
        :param y: y轴坐标
        :param z: 由 param x与 param y对应的值，
        Note:x=[[1,2,3]] y=[[1,2,3]]
        """
        self.x_list = x
        self.y_list = y
        if not isinstance(self.x_list[0], list) and not isinstance(self.x_list[0], np.ndarray):
            self.x_list = [self.x_list]
        if not isinstance(self.y_list[0], list) and not isinstance(self.y_list[0], np.ndarray):
            self.y_list = [self.y_list]
        self.name = "Line"
        self.title = "Line"
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

    def set_params(self, name, value):
        self.__setattr__(name, value)
    def get_object_name(self):
        return "scatter"
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

    def show(self, mode="markers", dash="no", line_shape="spline",use_colorscale=False):
            """
            :dash is dash dot dashdot
            :mode is markers lines markers+lines
            :line_shape is spline linear hv vh vhv hvh
            """
            if dash=="no":
                dash=None
            self.data = []
            iter = 0
            for x, y in zip(self.x_list, self.y_list):
                if use_colorscale==False:
                    if mode == "markers":
                        base_data = go.Scatter(x=x, y=y, name=self.name + str(iter), mode="markers", marker=
                        dict(color=init.get_color_list(y, self.r, self.g, self.b), size=init.get_size_list(y),), )
                    elif mode == "lines":
                        base_data = go.Scatter(x=x, y=y, name=self.name + str(iter), mode="lines", line=
                        dict(color=init.get_color(0, self.r, self.g, self.b, 0, 0), width=self.width, dash=dash),
                                               line_shape=line_shape,
                                               # fill='toself',
                                               # fillcolor='rgba(0,176,246,0.1)',
                                               )
                    else:
                        base_data = go.Scatter(x=x, y=y, name=self.name + str(iter), mode="lines+markers", line=
                        dict(color=init.get_color(0, self.r, self.g, self.b, 0, 0), width=self.width, dash=dash, ),
                                               line_shape=line_shape,
                                               # fill='toself',
                                               # fillcolor='rgba(0,176,246,0.1)',
                                               marker=
                                               dict(color=init.get_color_list(y, self.r, self.g, self.b),  size=init.get_size_list(y),
                                                    colorscale="Viridis",
                                                    showscale=True,),
                                               )
                else:
                    if mode == "markers":
                        base_data = go.Scatter(x=x, y=y, name=self.name + str(iter), mode="markers", marker=
                        dict(color=y, size=init.get_size_list(y),
                             colorscale="Blackbody",
                             showscale=True), )
                    elif mode == "lines":
                        base_data = go.Scatter(x=x, y=y, name=self.name + str(iter), mode="lines", line=
                        dict(color=init.get_color(0, self.r, self.g, self.b, 0, 0), width=self.width, dash=dash),
                                               line_shape=line_shape,
                                               # fill='toself',
                                               # fillcolor='rgba(0,176,246,0.1)',
                                               )
                    else:
                        base_data = go.Scatter(x=x, y=y, name=self.name + str(iter), mode="lines+markers", line=
                        dict(color=init.get_color(0, self.r, self.g, self.b, 0, 0), width=self.width, dash=dash, ),
                                               line_shape=line_shape,
                                               # fill='toself',
                                               # fillcolor='rgba(0,176,246,0.1)',
                                               marker=
                                               dict(color=init.get_color_list(y, self.r, self.g, self.b),
                                                    size=y,),
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
        x=self.x_list[0]
        y=self.y_list[0]
        mode = "markers"
        dash = "dot"
        line_shape = "spline"
        use_colorscale = False
        if hasattr(self,"mode"):
            mode=self.mode
        if hasattr(self,"dash"):
            dash=self.dash
        if hasattr(self,"line_shape"):
            line_shape=self.line_shape
        if hasattr(self,"use_colorscale"):
            use_colorscale=self.use_colorscale
        if use_colorscale == False:
            if mode == "markers":
                base_data = go.Scatter(x=x, y=y, name=self.name + str(iter), mode="markers", marker=
                dict(color=init.get_color_list(y, self.r, self.g, self.b), size=init.get_size_list(y), ),)
            elif mode == "lines":
                base_data = go.Scatter(x=x, y=y, name=self.name + str(iter), mode="lines", line=
                dict(color=init.get_color(0, self.r, self.g, self.b, 0, 0), width=self.width, dash=dash),
                                       line_shape=line_shape,
                                       # fill='toself',
                                       # fillcolor='rgba(0,176,246,0.1)',

                                       )
            else:
                base_data = go.Scatter(x=x, y=y, name=self.name + str(iter), mode="lines+markers", line=
                dict(color=init.get_color(0, self.r, self.g, self.b, 0, 0), width=self.width, dash=dash, ),
                                       line_shape=line_shape,
                                       # fill='toself',
                                       # fillcolor='rgba(0,176,246,0.1)',
                                       marker=
                                       dict(color=init.get_color_list(y, self.r, self.g, self.b),
                                            size=init.get_size_list(y),
                                            colorscale="Viridis",
                                            showscale=True, ),

                                       )
        else:
            if mode == "markers":
                base_data = go.Scatter(x=x, y=y, name=self.name + str(iter), mode="markers", marker=
                dict(color=y, size=init.get_size_list(y),
                     colorscale="Blackbody",
                     showscale=True),
              )
            elif mode == "lines":
                base_data = go.Scatter(x=x, y=y, name=self.name + str(iter), mode="lines", line=
                dict(color=init.get_color(0, self.r, self.g, self.b, 0, 0), width=self.width, dash=dash),
                                       line_shape=line_shape,
                                       # fill='toself',
                                       # fillcolor='rgba(0,176,246,0.1)',

                                       )
            else:
                base_data = go.Scatter(x=x, y=y, name=self.name + str(iter), mode="lines+markers", line=
                dict(color=init.get_color(0, self.r, self.g, self.b, 0, 0), width=self.width, dash=dash, ),
                                       line_shape=line_shape,
                                       # fill='toself',
                                       # fillcolor='rgba(0,176,246,0.1)',
                                       marker=
                                       dict(color=init.get_color_list(y, self.r, self.g, self.b),
                                            size=y, ),

                                       )
        return base_data
# x=np.linspace(0,100,100)
# y=np.random.randint(1,10,(100))
# Line([x],[y]).show(mode="markers")
