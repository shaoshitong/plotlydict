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


class Table(object):
    def __init__(self, data):
        """
        :param data: like [[["name1","namw2","name3"],[value1,value2,value3],...,[value1],[value2],[value3]]]
        其中[["name1","namw2","name3"],[value1,value2,value3],...,[value1],[value2],[value3]]象征一组数据
        """
        self.data_list = data
        if not isinstance(self.data_list[0], list):
            self.data_list = [self.data_list]
        self.name = "Table"
        self.title = "Table"
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
        self.align = "left"

    def get_object_name(self):
        return "table"

    def set_layout(self):
        self.layout = dict(
            title=self.title,
            font=dict(
                family=self.family,
                size=self.font_size,
                color=self.font_color,
            ),
            showlegend=False,
            autosize=False,
            height=600,
            width=800,
            legend=dict(
                x=self.legend_x,
                y=self.legend_y,
                bgcolor=self.legend_bgcolor,
                bordercolor=self.legend_bordercolor,
                borderwidth=self.borderwidth)
        )

    def set_params(self, name, value):
        self.__setattr__(name, value)

    def show(self):
        self.data = []
        iter = 0

        for data in self.data_list:
            base_data = go.Table(
                header=dict(
                    font=dict(size=self.font_size),
                    align=self.align,
                    values=data[0],
                ),
                name=self.name,
                cells=dict(
                    values=np.array(data[1:], dtype=np.float).transpose((1, 0)).tolist(),
                    align="left")
            )
            iter += 1
            self.data.append(base_data)
        self.set_layout()
        fig = Figure(data=self.data, layout=self.layout)
        fig.show()

    def get_fig(self):
        data = self.data_list[0]
        base_data = go.Table(
            header=dict(
                font=dict(size=self.font_size),
                align=self.align,
                values=data[0],
            ),
            name=self.name,
            cells=dict(
                values=np.array(data[1:], dtype=np.float).transpose((1, 0)).tolist(),
                align="left")
        )
        return base_data
#
# import numpy as np
# data=[["a","b","c"],[100,100,100]]
# Table([data]).show()
#
