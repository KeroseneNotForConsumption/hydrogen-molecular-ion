import os

import numpy as np

import plotly.graph_objects as go
import plotly.io as pio

if __name__ == 'plot_scatterplot':
    pio.renderers.default = 'jupyterlab'
else:
    pio.renderers.default = 'browser'

mos = [
    '1 sigma g',
    '1 sigma u *',
    '1 pi u',
    '2 sigma g',
    '2 sigma u *',
    '3 sigma g',
    '1 pi g *',
    '3 sigma u *',
]

# n, l, mu - same as the n, l, abs(m) 
# of the AO of the united atom limit.
mo_nums = {
    '1 sigma g': (1, 0, 0),
    '1 sigma u *': (2, 1, 0),
    '1 pi u': (2, 1, 1),
    '2 sigma g': (2, 0, 0),
    '2 sigma u *': (3, 1, 0),
    '3 sigma g': (3, 2, 0),
    '1 pi g *': (3, 2, 1),
    '3 sigma u *':(4, 3, 0),
}

mo_latex = {
    '1 sigma g': r"$1 \sigma_{g}$",
    '1 sigma u *': r"$1 \sigma_{u}^{\ast}$",
    '1 pi u': r"$1 \pi_{u}$",
    '2 sigma g' : r"$2 \sigma_{g}$",
    '3 sigma g' : r"$3 \sigma_{g}$",
    '2 sigma u *' : r"$2 \sigma_{u}^{*}$",
    '1 pi g *' : r"$1 \pi_{g}^{*}$",
    '3 sigma u *' : r"$3 \sigma_{u}^{*}$",
}

def plot_scatterplot():
    opacity = 0.25
    sample_size = 2000
    
    # y, z, x
    fig = go.Figure()
    
    fig.update_layout(
        template='plotly_white',
        margin=dict(l=5, r=5, t=5, b=5),
        height = 600,
        showlegend=False,
        scene = dict(
            xaxis_visible=False,
            yaxis_visible=False,
            zaxis_visible=False,
        )
    )
    
    # plot dots for the two nuclei
    fig.add_trace(
        go.Scatter3d(
            x=[0, 0],
            y=[-1, 1],
            z=[0, 0],
            mode='markers',
            marker=dict(
                size=2,
                color='black',
                opacity=1.0
            )
        )
    )
    
    buttons = []
    
    for ind, mo in enumerate(mos):
        n, l, mu = mo_nums[mo]
        
        # retrieve data from file
        filename = f"samples_{n}{l}{mu}.npy"
        filedir = os.path.join(os.path.abspath(''), 
                               'data',
                               'scatterplot',
                               filename)
        samples_loaded = np.load(filedir)
        
        # this is how the data is structured in the file
        points = samples_loaded[:sample_size, :3]
        points_sign = samples_loaded[:sample_size, 3].astype(bool)
    
        pos_points = points[points_sign, :]
        neg_points = points[np.logical_not(points_sign), :]
        
        # only 1 sigma g visible at startup
        init_visible = True if ind == 0 else False
        
        fig.add_trace(
            go.Scatter3d(
                visible=init_visible,
                x=pos_points[:, 1], 
                y=pos_points[:, 2], 
                z=pos_points[:, 0],
                mode='markers',
                marker=dict(
                    size=5,
                    color='red',                
                    opacity=opacity
                )
            )
        )
        
        if n + mu + l > 1: #excludes 1 sigma g
            fig.add_trace(
                go.Scatter3d(
                    visible=init_visible,
                    x=neg_points[:, 1], 
                    y=neg_points[:, 2], 
                    z=neg_points[:, 0],
                    mode='markers',
                    marker=dict(
                        size=5,
                        color='blue',                
                        opacity=opacity
                    )
                )
            )
        
        button = dict(
            label=mo_latex[mo],
            method='update',
            args=[{'visible': [False] * 17}])
        
        # 1. make the nuclei visible for all options
        # 2. make the correct scatterplot (positive or negative)
        # for each choice
        button['args'][0]['visible'][0] = True
        button['args'][0]['visible'][2*ind] = True
        button['args'][0]['visible'][2*ind+1] = True
        buttons.append(button)
    
    
        
    fig.update_traces(
        hoverinfo='skip',
        showlegend=False
    )
        
    fig.update_layout(
        updatemenus=[
            dict(
                active=0,
                buttons=buttons,
                direction="down",
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.01,
                xanchor="left",
                y=1.1,
                yanchor="top"
            )
        ]
    )
    
    
    
    fig.update_layout(
        scene = dict(
            xaxis = dict(range=[-24, 24]),
            yaxis = dict(range=[-40, 40]),
            zaxis = dict(range=[-24, 24]),
            aspectratio=dict(x=24, 
                             y=40, 
                             z=24,),
            camera=dict(eye=dict(x=25, y=0.0, z=0.0))))
    
    
    fig.show()

if __name__ == '__main__':
    plot_scatterplot()