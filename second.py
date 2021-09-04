import numpy as np
import plotly as py
import chart_studio
import plotly.io as pio
import plotly.express as px

if __name__=="__main__":
    pio.templates.default = "simple_white"
    pyplt = py.offline.plot
    p = pio.renderers['png']
    p.width = 800
    p.height = 600
    chart_studio.tools.set_config_file(world_readable=True, sharing='public')

    df = px.data.iris()
    print(df)
    fig = px.scatter_3d(df, x='sepal_length', y='sepal_width', z='petal_width',
                        color_continuous_midpoint=0.5,
                        color='petal_length', symbol='species')
    fig.show()