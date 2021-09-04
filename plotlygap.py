import json
import pandas as pd
import plotly.graph_objects as go
import plotly as py
import chart_studio
import plotly.io as io
import numpy as np
pyplt = py.offline.plot
p=io.renderers['png']
p.width=800
p.height=600
import openpyxl
from plotly.graph_objs import *
from plotly.graph_objs.layout import *
chart_studio.tools.set_config_file(world_readable=True,
                             sharing='public')
china_ncp = pd.read_excel('ncp_map.xlsx',engine='openpyxl')
china_ncp.head()
with open(r'china.json', encoding='utf-8')as f:
    china_geo = json.load(f)
##token = 'pk.eyJ1Ijoic3l2aW5jZSIsImEiOiJjazZrNTcwY3kwMHBrM2txaGJqZWEzNWExIn0.tLQHY_OoiR2NMxnYHXUBAA'
figure =dict(type='choropleth',geojson=china_geo, locations=china_ncp['FIPS'], z=china_ncp['province_confirmedCount'],
                        name='NCP',
                        hovertext=china_ncp['Provinces'], hoverinfo='text+z', zmax=2000, zmin=0,
                        text=china_ncp['province_curedCount'],
                        colorscale='YlOrRd', marker_line_width=0.5, marker_line_color='rgb(169,164,159)',hovertemplate = '<b>Province</b>: <b>%{hovertext}</b>' + \
                            '<br> <b>确诊人数 </b>: %{z}<br>' + \
                            '<br> <b>治愈人数 </b>: %{text}<br>')
figure2=go.Scatter(x=np.linspace(1,10,1000),y=np.random.randn(1000),mode = 'markers',name = 'pred',
    marker = dict(
        size = 16,
        color = np.random.randn(1000),
        colorscale ='peach',
        showscale = True
    ))
update_layout=dict(
                  title={'text': '疫情地图', 'xref': 'paper', 'x': 0.5},
                  margin={'l': 10, 'r': 0, 't': 50, 'b': 10})
layout = dict(
    template="plotly_dark",

)
data=[figure2]
fig=Figure(data=data,layout=layout)
fig.show()