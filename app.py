############################## Memory State Test ##############################
import resource

def mem():
    print('### Memory usage         : % 2.2f MB ###' % round(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.0,1)
    )
###############################################################################

print("Initial M.S.:")
mem()

# Dash libs
import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go

print("After Dash Imports M.S.:")
mem()

# Scipy libs
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import cm

print("After Scipy Imports M.S.:")
mem()

# My modules
import setup
from compute_memsave import compute
from utils import DotDict, cmap_to_cscale
from meccolormap import jet_white_r
from termomecanico_memsave import x_axis
from termomecanico_memsave import y_axis
from termomecanico_memsave import z_axis
from termomecanico_memsave import xy_step
from termomecanico_memsave import grid_2D
from termomecanico_memsave import topo
from termomecanico_memsave import icd
from termomecanico_memsave import moho
from termomecanico_memsave import slab_lab
from termomecanico_memsave import geotherm
from termomecanico_memsave import yse
from termomecanico_memsave import shf
from termomecanico_memsave import eet

def get_x_axis():
    return x_axis
def get_y_axis():
    return y_axis
def get_z_axis():
    return z_axis
def get_xy_step():
    return xy_step
def get_x_grid_2D():
    return grid_2D[0]
def get_y_grid_2D():
    return grid_2D[1]
def get_topo():
    return topo
def get_icd():
    return icd
def get_moho():
    return moho
def get_slab_lab():
    return slab_lab
def get_geotherm():
    return geotherm
def get_tension_yse():
    return yse[0]
def get_compression_yse():
    return yse[1]
def get_surface_heat_flow():
    return shf
def get_effective_elastic_thickness():
    return eet

print("After My Modules Imports M.S.:")
mem()

print("After All Module Imports M.S.:")
mem()

###############################################################################
########################## Data Manipulation / Model ##########################
###############################################################################

## Static Input
#gm_data = np.loadtxt('data/Modelo.dat')
#areas = np.loadtxt('data/areas.dat')
#trench_age = np.loadtxt('data/PuntosFosaEdad.dat')
#rhe_data = setup.read_rheo('data/Rhe_Param.dat')
#
## User Input
#t_input = setup.readVars('VarTermal.txt')
#m_input = setup.readVars('VarMecanico.txt')
#
#print("After Input M.S.:")
#mem()
#
## Models Generation
#def compute_models(gm_data, areas, trench_age, rhe_data,
#                   t_input=None, m_input=None):
#    if t_input is not None and m_input is not None:
#        pass
#    else:
#        t_input, m_input = read_user_input()
#    models_values = compute(gm_data, areas, trench_age, rhe_data,
#                            t_input, m_input)
#    models_keys = ['D', 'CS', 'GM', 'TM', 'MM']
#    models = dict(zip(models_keys, models_values))
#    return DotDict(models)
#
#models = compute_models(gm_data, areas, trench_age, rhe_data, t_input, m_input)

print("After Models M.S.:")
mem()

# Models Wrapper Methods
#def get_x_axis():
#    return models.CS.get_x_axis()
#def get_y_axis():
#    return models.CS.get_y_axis()
#def get_z_axis():
#    return models.CS.get_z_axis()
#def get_xy_step():
#    return models.CS.get_xy_step()
#def get_x_grid_2D():
#    return models.CS.get_2D_grid()[0]
#def get_y_grid_2D():
#    return models.CS.get_2D_grid()[1]
#def get_topo():
#    return models.GM.get_topo().mask_irrelevant()
#def get_icd():
#    return models.GM.get_icd().mask_irrelevant()
#def get_moho():
#    return models.GM.get_moho().mask_irrelevant()
#def get_slab_lab():
#    return models.GM.get_slab_lab().mask_irrelevant()
#def get_geometry():
#    return models.GM.get_3D_geometric_model()
#def get_geotherm():
#    return models.TM.get_geotherm()
#def get_tension_yse():
#    return models.MM.get_yse()[0]
#def get_compression_yse():
#    return models.MM.get_yse()[1]
#def get_surface_heat_flow():
#    return models.TM.get_surface_heat_flow()
#def get_effective_elastic_thickness():
#    return models.MM.get_eet()
def map_grids():
    map_grids = {
        'None': None,
        'Surface Heat Flow': get_surface_heat_flow(),
        'Effective Elastic Thickness': get_effective_elastic_thickness()}
    return map_grids
def cross_section_grids():
    cross_section_grids = {
        'None': None,
        'Geotherm': get_geotherm(),
        'Yield Strength Envelope (T)': get_tension_yse(),
        'Yield Strength Envelope (C)': get_compression_yse()}
    return cross_section_grids

# Plotly color palettes
def coolwarm_palette():
    coolwarm_cmap = cm.get_cmap('coolwarm')
    coolwarm_cscale = cmap_to_cscale(coolwarm_cmap, 255) 
    return coolwarm_cscale
def afmhot_palette():
    afmhot_r_cmap = cm.get_cmap('afmhot_r')
    afmhot_r_cscale = cmap_to_cscale(afmhot_r_cmap, 255) 
    return afmhot_r_cscale
def jet_palette():
    jet_white_r_cmap = jet_white_r
    jet_white_r_cscale = cmap_to_cscale(jet_white_r_cmap, 255)
    return jet_white_r_cscale

# Visualization related methods
def choose_map_colorscale(map_grid): 
    if map_grid == 'Surface Heat Flow':
        color_palette = afmhot_palette()
        color_limits = {'min': None, 'max': None}
    elif map_grid == 'Effective Elastic Thickness':
        color_palette = jet_palette()
        color_limits = {'min': 0, 'max': 100}
    else:
        color_palette = None
        color_limits = {'min': None, 'max': None}
    return {'color_palette': color_palette, 'color_limits': color_limits}
def choose_cross_section_colorscale(cross_section_grid): 
    if cross_section_grid == 'Geotherm':
        color_palette = coolwarm_palette()
        color_limits = {'min': 0, 'max': 1300}
    elif cross_section_grid == 'Yield Strength Envelope (T)':
        color_palette = jet_palette()
        color_limits = {'min': 0, 'max': 200}
    elif cross_section_grid == 'Yield Strength Envelope (C)':
        color_palette = jet_palette()
        color_limits = {'min': 0, 'max': 200}
    else:
        color_palette = None
        color_limits = {'min': None, 'max': None}
    return {'color_palette': color_palette, 'color_limits': color_limits}
def plot_map_data(latitude, map_grid):
    colorscale = choose_map_colorscale(map_grid)
    grid = map_grids()[map_grid]
    if map_grid == 'None':
        marker = {'opacity': 0.2}
    else:
        marker = (
            {'color': grid.flatten(),
            'colorscale': colorscale['color_palette'],
            'colorbar': {},
            'cmin': colorscale['color_limits']['min'],
            'cmax': colorscale['color_limits']['max'],
            'line': {'color': 'rgba(0,0,0,0)'},
            'opacity': 1})
    return ( 
        [go.Scattergeo(
            lon = get_x_grid_2D().flatten(),
            lat = get_y_grid_2D().flatten(),
            marker = marker,
            showlegend = False,
            geo = 'geo'),
        go.Scattergeo(
            lon = [-82.5, -57.5],
            lat = latitude,
            mode='lines',
            line= {'width':1, 'color':'black'},
            showlegend = False)]
    )

def plot_cross_section_data(latitude, cross_section_grid):
    colorscale = choose_cross_section_colorscale(cross_section_grid)
    grid = cross_section_grids()[cross_section_grid]
    #index = np.where(get_y_axis() == latitude)[0][0]
    #print(index)
    #print(cross_section_grid)
    #print(type(grid))
    if cross_section_grid == 'None':
        heatmap = go.Heatmap()
        znan = None
    else:
        heatmap = go.Heatmap(
            x = get_x_axis(),
            y = get_z_axis(),
            z = grid.cross_section(latitude=latitude).T,
            #z = grid[:, index, :].T,
            #z = z,
            colorscale = colorscale['color_palette'],
            zmin = colorscale['color_limits']['min'],
            zmax = colorscale['color_limits']['max'],
            zsmooth = 'best',
            #name = name
        )
        znan = grid.cross_section(latitude=latitude).T.copy()
        #znan = grid[:, index, :].T.copy()
        invalid = np.isnan(znan)
        znan[invalid] = 1
        znan[np.invert(invalid)] = None
    return (
        [heatmap,
        go.Scatter(
            x=get_x_axis(),
            y=get_topo().cross_section(latitude=latitude),
            #y=get_topo()[:, index].T,
            name='topo'),
        go.Scatter(
            x=get_x_axis(),
            y=get_icd().cross_section(latitude=latitude),
            #y=get_icd()[:, index].T,
            name='icd'),
        go.Scatter(
            x=get_x_axis(),
            y=get_moho().cross_section(latitude=latitude),
            #y=get_moho()[:, index].T,
            name='moho'),
        go.Scatter(
            x=get_x_axis(),
            y=get_slab_lab().cross_section(latitude=latitude),
            #y=get_slab_lab()[:, index].T,
            name='slab/lab'),
        go.Heatmap(
            x = get_x_axis(),
            y = get_z_axis(),
            z = znan,
            showscale = False,
            hoverinfo = 'skip')]
    )

print("After Models Variables M.S.:")
mem()

###############################################################################
########################### Dashboard Layout / View ###########################
###############################################################################

app = dash.Dash()

app.css.append_css({"external_url": "https://codepen.io/chriddyp/pen/bWLwgP.css"})

app.config.suppress_callback_exceptions = True

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

# Pages layouts
def index_layout(): 
    return html.Div('Index')

def maker_layout(): 
    return html.Div('Maker')

def explorer_layout():
    return html.Div([
        'Explorer',
        html.Div(map_layout(), 
                 className='five columns'),
        html.Div(cross_section_layout(), 
                 className='seven columns')#, 
                 #style={'marginLeft': '0'})
    ])

# Sections layouts
def map_layout(): 
    return [
        #
        html.Div([
            dcc.Graph(id='map'),
            dcc.RadioItems(
                id='map_grid_options',
                options=map_grid_options(),
                value=list(map_grid_options()[0].values())[0],
                labelStyle={'display': 'inline-block'}
            )
        ], className='ten columns'),
        #
        html.Div([
            dcc.Slider(
                id='latitudes',
                max=map_maximum_latitude(),
                min=map_minimum_latitude(),
                marks=map_latitude_marks(),
                value=list(map_latitude_marks().keys())[0],
                step=map_latitude_step(),
                vertical=True
            ),
        ], className='two columns', style={'height': '720', 'paddingTop': '120'})
    ]

def cross_section_layout(): 
    return [
        html.Div([
            dcc.RadioItems(
                id='cross_section_grid_options',
                options=cross_section_grid_options(),
                value=list(cross_section_grid_options()[0].values())[0],
                labelStyle={'display': 'inline-block'}
            )
        ], style={'paddingLeft': 100}),
        dcc.Graph(id='cross_section'),
    ]

def yse_chart_layout():
    return [dcc.Graph(id='yse')]

# Appareance Methods
def map_grid_options():
    grid_options = (
        [{'label': grid, 'value': grid}
        for grid in list(map_grids().keys())])
    return grid_options
def map_latitude_marks():
    latitude_marks = (
        {int(i): '{}'.format(i) 
        for i in get_y_axis()[::10]})
    return latitude_marks
def map_maximum_latitude():
    return max(get_y_axis())
def map_minimum_latitude():
    return min(get_y_axis())
def map_latitude_step():
    return get_xy_step()
def cross_section_grid_options():
    grid_options = (
        [{'label': grid, 'value': grid}
        for grid in list(cross_section_grids().keys())])
    return grid_options
def map_graph_layout():
    return ( 
        {'title': 'Mapa',
        'autosize': False,
        'height': 800,
        'width': 600,
        'margin': {'l':0,'r':0,'t':0,'b':0,'pad':2,'autoexpand':True},
        'geo': {
            'scope': 'south america',
            'resolution': 50,
            'showframe': True,
            'showcoastlines': True,
            'coastlinecolor': 'rgb(0, 0, 0)',
            'coastlinewidth':2,
            'showland': True,
            'showocean':True,
            'showcountries': False,
            'projection': {
                'type': 'mercator',
            },
            'lonaxis': {
                'range':  [-81.0, -59.0],
                'showgrid': False,
                'dtick': 5
            },
            'lataxis': {
                'range': [-47.0, -9.0],
                'showgrid': False,
                'dtick': 5,
                'gridwidth': 2
            },
        }}
    )

def cross_section_graph_layout():
    return (
        {'legend': {'orientation':'h'},
        'margin': {'l':0,'r':0,'t':0,'b':0,'pad':2,'autoexpand':True},
        'xaxis': {'range': [-81.0, -59.0]},
        'yaxis': {'range': [-180.0, 10.0]}
        }
    )
  


print("After Layout M.S.:")
mem()

###############################################################################
################# Interaction Between Components / Controller #################
###############################################################################

@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/':
        return index_layout()
    if pathname == '/maker':
        return maker_layout() 
    elif pathname == '/explorer':
        return explorer_layout()
    else:
        return '404'

@app.callback(
    Output('map', 'figure'),
    [Input('latitudes', 'value'), Input('map_grid_options', 'value')])
def update_map(latitude, map_grid):
    print("update_map called")
    data = plot_map_data(latitude, map_grid)
    layout = map_graph_layout()
    figure = {'data': data, 'layout': layout}
    return figure

@app.callback(
    Output('cross_section', 'figure'),
    [Input('latitudes', 'value'), Input('cross_section_grid_options', 'value')]
)
def update_cross_section(latitude, cross_section_grid):
    print("update_cross_section_called")
    data = plot_cross_section_data(latitude, cross_section_grid)
    layout = cross_section_graph_layout()
    figure = {'data': data, 'layout': layout}
    return figure


print("After Controller M.S.:")
mem()

###############################################################################

if __name__ == '__main__':
    app.run_server(debug=True)

"""

app.layout = html.Div([


], className='twelve columns')

def get_map_model(map_model_value):
    if map_model_value == 'shf':
        map_model = shf
        colorscale = afmhot_r
        color_limits = [np.nanmin(shf), np.nanmax(shf)]
    elif map_model_value == 'eet':
        map_model = eet
        colorscale = jet
        color_limits = [0, 100]
    else:
        map_model = None
        colorscale = None
        color_limits = None

    return map_model, colorscale, color_limits

def get_model(model_value):
    if model_value == 'gt':
        model = gt
        colorscale = coolwarm
        color_limits = [0, 1300]
        name = 'temperature'
    elif model_value == 'yse_t':
        model = yse_t
        colorscale = jet
        color_limits = [0, 200]
        name = 'yield strength (t)'
    elif model_value == 'yse_c':
        model = yse_c
        colorscale = jet
        color_limits = [0, 200]
        name = 'yield strength (c)'
    else:
        model = None
        colorscale = None
        color_limits = None
        name = None
    return model, colorscale, color_limits, name


"""
"""
@app.callback(
    Output('yse', figure),
    [Input('map', clickData), Input('cross-section', clickData)]
)
def update_yse(map_click, cs_click):
    x = map_click['points'][0]['x']
    y = map_click['points'][0]['y']
    return {
        'data': [
            go.Scatter(
"""
"""
"""
