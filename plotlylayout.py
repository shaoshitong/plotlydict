import json
import math
import numpy as np
import plotly.graph_objects as go
import plotly as py
import chart_studio
import plotly.io as io
import plotly.io as pio
pyplt = py.offline.plot
p=io.renderers['png']
p.width=800
p.height=600
import openpyxl
from plotly.graph_objs import *
from plotly.graph_objs.layout import *
chart_studio.tools.set_config_file(world_readable=True,
                             sharing='public')
pio.templates.default = "none"
data1=go.Bar(
            x = ["Cars", "Pedestrians", "Cyclists"],
            y = [80.64, 54.64, 70.03],
            name = "PointRCNN",
            marker_color='rgb(255, 100, 100)'
    )
data2=go.Bar(
            x = ["Cars", "Pedestrians", "Cyclists"],
            y = [82.91, 59.67, 70.13],
            name = "PartA^2",
marker_color='rgb(255,0, 100)'
    )
data3=go.Bar(
            x = ["Cars", "Pedestrians", "Cyclists"],
            y = [80.36, 54.49, 70.38],
            name = "PV-RCNN",
marker_color='rgb(100, 255, 255)'
    )
data4=go.Bar(
            x = ["Cars", "Pedestrians", "Cyclists"],
            y = [78.39, 51.41, 62.93],
            name = "PointPillar",
marker_color='rgb(100, 255, 100)'
    )
data5=go.Bar(
            x = ["Cars", "Pedestrians", "Cyclists"],
            y = [81.61, 51.14, 66.74],
            name = "SECOND",
marker_color='rgb(100, 100, 255)'
    )
data6=go.Bar(
            x = ["Cars", "Pedestrians", "Cyclists"],
            y = [83.01, 58.14, 71.24],
            name = "Ours-(GAT+PointNet++)",
marker_color='rgb(255,200, 120)'
    )
layout=dict(
    title="相关算法准确率对比图",  # 标题文本 不设置位置的话 默认在左上角，下面有设置位置和颜色的讲解
    showlegend=True,
    legend_title="Legend",  # 图例标题文本
    # 设置图例相对于左下角的位置
    legend=dict(
        x=0.9,
        y=1.1
    ),
    font=dict(
        family="Segoe UI Black",  # 所有标题文字的字体
        size=16,  # 所有标题文字的大小
        color="black"  # 所有标题的颜色
    ),
)
data=[data1,data2,data3,data4,data5,data6]
fig=Figure(data=data,layout=layout)
fig.show()