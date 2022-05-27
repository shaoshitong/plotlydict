import sys, os


def add_init():
    root = os.path.dirname(os.path.realpath(__file__))
    if root not in sys.path:
        sys.path.append(root)


from plotly.subplots import make_subplots

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


class Subplot(object):
    def __init__(self, Object):
        """
        :param Object: select from Bar,Contour,Histogram2D,Line,Mash3D,Pie,Redar,Surface,Table
        """
        assert (isinstance(Object, list))
        for obj in Object:
            assert isinstance(obj, object)
        self.len = len(Object)
        self.object = Object
        self.l = self._get_l1_l2()
        self.name = "Subplot"
        self.list = [[1 for j in range(self.l[0])] for i in range(self.l[1])]
        iter = 0
        for i in range(len(self.list[0])):
            for j in range(len(self.list)):
                self.list[j][i] = {'type': self.object[iter].get_object_name()}
                iter += 1

        self.fig = make_subplots(
            rows=self.l[1], cols=self.l[0],
            specs=self.list
            , subplot_titles=[obj.name for obj in Object],
        )

    def _zys(self, n):
        value = []
        i = 2
        m = n
        while i <= int(m / 2 + 1) and n != 1:
            if n % i == 0:
                value.append(i)
                n = n // i
                i -= 1
            i += 1
        value.append(1)
        if len(value) == 1:
            value.append(m)
        return value

    def _get_l1_l2(self):
        l = self._zys(self.len)
        l1 = 1
        l2 = 1
        tag = 0
        for i in l:
            if tag:
                l1 *= i
                tag = 0
            else:
                l2 *= i
                tag = 1
        return l1, l2

    def set_layout(self):
        self.fig.update_layout(
            title_text=self.name,
            height=800,
            width=2000,
            legend=dict(
                x=1.1,
                y=1.3,
                bgcolor="white",
                bordercolor="black",
                borderwidth=2)
        )

    def show(self):
        l1, l2 = self.l
        l_all = l1 * l2
        iter = 0
        iter_l1, iter_l2 = self._set_iter(l2, iter)
        while iter < l_all:
            self.fig.add_trace(self.object[iter].get_fig(), row=iter_l2 + 1, col=iter_l1 + 1)
            iter += 1
            iter_l1, iter_l2 = self._set_iter(l2, iter)

        self.set_layout()
        self.fig.show()

    def _set_iter(self, m, iter):
        return int(iter // m), int(iter % m)
