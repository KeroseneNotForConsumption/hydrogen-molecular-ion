import os

import numpy as np

import plotly.graph_objects as go
import plotly.io as pio

if __name__ == 'plot_isosurface':
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

mo_notation = {
    '1 sigma g': '1σg',
    '1 sigma u *': '1σu*',
    '1 pi u': '1πu',
    '2 sigma g' : '2σg',
    '3 sigma g' : '3σg',
    '2 sigma u *' : '2σu*',
    '1 pi g *' : '1πg*',
    '3 sigma u *' : '3σu*',
}


def plot_isosurface():
    opacity = 0.6
    
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
        
        # retrieve verts data from file
        filename = f"isosurface_{n}{l}{mu}_pos_verts.npy"
        filedir = os.path.join(os.path.abspath(''), 
                               'data',
                               'isosurface',
                               filename)
        verts_pos = np.load(filedir)
        
        # retrieve faces from file
        filename = f"isosurface_{n}{l}{mu}_pos_faces.npy"
        filedir = os.path.join(os.path.abspath(''), 
                               'data',
                               'isosurface',
                               filename)
        faces_pos = np.load(filedir)
        
        # only 1 sigma g visible at startup
        init_visible = True if ind == 0 else False
        
        fig.add_trace(
            go.Mesh3d(
                x=verts_pos[:, 1],
                y=verts_pos[:, 2],
                z=verts_pos[:, 0],
                i = faces_pos[:, 1],
                j = faces_pos[:, 2],
                k = faces_pos[:, 0],
                color='red',
                opacity=opacity,
                visible=init_visible
            )
        )
        
        if n + mu + l > 1: # excludes 1 sigma g
            # retrieve verts data from file
            filename = f"isosurface_{n}{l}{mu}_neg_verts.npy"
            filedir = os.path.join(os.path.abspath(''), 
                                   'data',
                                   'isosurface',
                                   filename)
            verts_neg = np.load(filedir)
            
            # retrieve faces from file
            filename = f"isosurface_{n}{l}{mu}_neg_faces.npy"
            filedir = os.path.join(os.path.abspath(''), 
                                   'data',
                                   'isosurface',
                                   filename)
            faces_neg = np.load(filedir)
            
            fig.add_trace(
                go.Mesh3d(
                    x=verts_neg[:, 1],
                    y=verts_neg[:, 2],
                    z=verts_neg[:, 0],
                    i = faces_neg[:, 1],
                    j = faces_neg[:, 2],
                    k = faces_neg[:, 0],
                    color='blue',
                    opacity=opacity,
                    visible=init_visible
                )
            )
        
        
        
        
        button = dict(
            label=mo_notation[mo],
            method='update',
            args=[{'visible': [False] * 16}])
        
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
    plot_isosurface()