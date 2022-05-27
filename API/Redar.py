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


def Line(line_style, line_color, line_width):
    """
    This function generate a plotly line object
    """
    Line = go.scatterpolar.Line(dash=line_style, color=line_color, width=line_width)
    return Line


class Redar(object):
    def __init__(self, labels, values):
        """
        :param labels: [["a","b","c","d"]]
        :param values: [[100,200,300,400]] or [[[100,200,300,400],[100,300,400,500]]]
        """
        self.x_list = labels
        self.y_list = values
        if not isinstance(self.x_list[0], list) and not isinstance(self.x_list[0], np.ndarray):
            self.x_list = [self.x_list]
        assert isinstance(self.y_list, list)
        if not isinstance(self.y_list[0], list):
            self.y_list = [self.y_list]
        self.name = "Redar chart"
        self.title = "Redar chart"
        self.family = "Courier New, monospace"
        self.line_close = True
        self.fill = "toself"
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
        self.hole = 0
        self.r = random.randint(0, 256)
        self.g = random.randint(0, 256)
        self.b = random.randint(0, 256)
        self.y_max = 0.
        for y in self.y_list:
            if not isinstance(y[0], list):
                y = [y]
            for m in y:
                p = max(m)
                if self.y_max < p:
                    self.y_max = p
        self.line_colors = [
            "rgba(255, 148, 17,0.9)",
            "rgba(255, 219, 25,0.9)",
            "rgba(25, 25, 150,0.9)",
            "rgba(169, 54, 241,0.9)",
            "rgba(176, 176, 176,0.9)",
            "rgba(20, 255, 255,0.9)",
        ]
        self.fillcolors = [
            "rgba(255, 148, 17,0.1)",
            "rgba(255, 219, 25,0.1)",
            "rgba(25, 25, 150,0.1)",
            "rgba(169, 54, 241,0.1)",
            "rgba(176, 176, 176,0.1)",
            "rgba(20, 255, 255,0.1)", ]

    def set_params(self, name, value):
        self.__setattr__(name, value)

    def get_object_name(self):
        return "scatterpolar"

    def set_layout(self):
        self.layout = dict(
            uniformtext_minsize=12,
            uniformtext_mode='hide',
            title=self.title,
            polar=dict(
                radialaxis=dict(
                    visible=False,
                    range=[0, self.y_max]
                )),
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
            plot_bgcolor='rgb(255, 255, 255)',
            legend=dict(
                x=self.legend_x,
                y=self.legend_y,
                bgcolor=self.legend_bgcolor,
                bordercolor=self.legend_bordercolor,
                borderwidth=self.borderwidth)
        )

    def show(self):
        self.data = []
        iter = 0
        for x, y in zip(self.x_list, self.y_list):
            if not isinstance(y[0], list):
                y = [y]
            for m in y:
                base_data = go.Scatterpolar(r=m, theta=x,
                                            fillcolor=self.fillcolors[random.randint(0, len(self.fillcolors) - 1)],
                                            showlegend=False,
                                            fill=self.fill,
                                            name=self.name,
                                            line=Line("dash",
                                                      self.line_colors[random.randint(0, len(self.line_colors) - 1)], 3)
                                            )
                self.data.append(base_data)
                iter += 1
        self.set_layout()
        fig = Figure(data=self.data, layout=self.layout)
        fig.show()

    def get_fig(self):
        x = self.x_list[0]
        y = self.y_list[0]
        iter = 0
        if not isinstance(y[0], list):
            y = [y]
        m = y[0]
        base_data = go.Scatterpolar(r=m, theta=x,
                                    fillcolor=self.fillcolors[random.randint(0, len(self.fillcolors) - 1)],
                                    showlegend=False,
                                    fill=self.fill,
                                    name=self.name,
                                    line=Line("dash", self.line_colors[random.randint(0, len(self.line_colors) - 1)], 3)
                                    )
        iter += 1
        return base_data