############################## Memory State Test ##############################
import resource
import os
import psutil
process = psutil.Process(os.getpid())

def mem():
    print('### Memory usage (resource)      : % 2.2f MB ###' % round(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.0,1)
    )
def mem2():
    print('### Memory usage (psutil)        : % 2.2f MB ###' % round(
        process.memory_info().rss/1.e6))

###############################################################################

print("Initial M.S.:")
mem()
mem2()

# Python libs
from textwrap import dedent as d
import gc
import pandas
import json
import jsonpickle
import jsonpickle.ext.numpy as jsonpickle_numpy
jsonpickle_numpy.register_handlers()
#import sys
#sys.setrecursionlimit(17900)

# Dash libs
import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
import idest_dash_components as idc
import plotly.graph_objs as go

print("After Dash Imports M.S.:")
mem()
mem2()

# Scipy libs
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import cm

print("After Scipy Imports M.S.:")
mem()
mem2()

# My modules
import setup
from compute import (
    compute,
    compute_data,
    compute_cs,
    compute_gm,
    compute_tm,
    compute_mm
)
from utils import DotDict, cmap_to_cscale
from meccolormap import jet_white, jet_white_r

print("After My Modules Imports M.S.:")
mem()
mem2()

print("After All Module Imports M.S.:")
mem()
mem2()

###############################################################################
########################## Data Manipulation / Model ##########################
###############################################################################

# Static Input
gm_data = np.loadtxt('data/Modelo.dat')
areas = np.loadtxt('data/areas.dat')
trench_age = np.loadtxt('data/PuntosFosaEdad.dat')
rhe_data = setup.read_rheo('data/Rhe_Param.dat')

# User Input
t_input = setup.readVars('VarTermal.txt')
m_input = setup.readVars('VarMecanico.txt')

# Earthquakes
edf = pandas.read_csv('earthquakes/1900_2018_09_15_25+.csv')
edf = edf[edf.depth < 200]
edf['text'] = edf['place'].astype(str)+'<br>'+\
    'lat: '+edf['latitude'].astype(str)+', lon: '+edf['longitude'].astype(str)+'<br>'+\
    'time: '+edf['time'].astype(str)+'<br>'+\
    'magnitude: '+edf['mag'].astype(str)+'<br>'+\
    'depth: '+edf['depth'].astype(str)+'<br>'+\
    'horizontal error: '+edf['horizontalError'].astype(str)+'<br>'+\
    'depth error: '+edf['depthError'].astype(str)

print("After Input M.S.:")
mem()
mem2()

# Input generation
def generate_input():
    return

# Models Generation
def compute_models(gm_data, areas, trench_age, rhe_data,
                   t_input=None, m_input=None):
    if t_input is not None and m_input is not None:
        pass
    else:
        t_input, m_input = read_user_input()
    models_values = compute(gm_data, areas, trench_age, rhe_data,
                            t_input, m_input)
    models_keys = ['D', 'CS', 'GM', 'TM', 'MM']
    models = dict(zip(models_keys, models_values))
    return DotDict(models)

models = compute_models(gm_data, areas, trench_age, rhe_data, t_input, m_input)


print("After Models M.S.:")
mem()
mem2()

#pickle = jsonpickle.encode(models.TM.get_geotherm().array)
#decoded_pickle = jsonpickle.decode(pickle)
#pickle = jsonpickle.encode(models.TM.get_surface_heat_flow().array)
#pickle = jsonpickle.encode(models.TM.get_geotherm().cross_section(latitude=-15).T)

print("########################After Pickle M.S.:")
mem()
mem2()

# Models Wrapper Methods
def get_x_axis():
    return models.CS.get_x_axis()
def get_y_axis():
    return models.CS.get_y_axis()
def get_z_axis():
    return models.CS.get_z_axis()
def get_xy_step():
    return models.CS.get_xy_step()
def get_x_grid_2D():
    return models.CS.get_2D_grid()[0]
def get_y_grid_2D():
    return models.CS.get_2D_grid()[1]
def get_topo():
    return models.GM.get_topo().mask_irrelevant()
def get_icd():
    return models.GM.get_icd().mask_irrelevant()
def get_moho():
    return models.GM.get_moho().mask_irrelevant()
def get_slab_lab():
    return models.GM.get_slab_lab().mask_irrelevant()
def get_boundaries():
    boundaries = {
        'topo': {'array': get_topo(), 'color': 'rgb(109,59,15)'},
        'icd': {'array': get_icd(), 'color': 'rgb(154,29,71)'},
        'moho': {'array': get_moho(), 'color': 'rgb(147,94,25)'},
        'slablab': {'array': get_slab_lab(), 'color': 'rgb(86,96,28)'}
    }
    return boundaries
def get_layers():
    layers = {
        'upper crust': {'color': 'rgb(219,151,163)'},
        'lower crust': {'color': 'rgb(224,187,139)'},
        'litospheric mantle': {'color': 'rgb(181,189,137)'},
    }
    return layers
def get_geometry():
    return models.GM.get_3D_geometric_model()
def get_geotherm(thermal_model):
    #return models.TM.get_geotherm()
    return thermal_model.get_geotherm()
def get_tension_yse(mechanical_model):
    #return models.MM.get_yse()[0]
    return mechanical_model.get_yse()[0]
def get_compression_yse(mechanical_model):
    #return models.MM.get_yse()[1]
    return mechanical_model.get_yse()[1]
def get_surface_heat_flow(thermal_model):
    #return models.TM.get_surface_heat_flow()
    return thermal_model.get_surface_heat_flow(format='positive milliwatts')
def get_effective_elastic_thickness(mechanical_model):
    #return models.MM.get_eet()
    return mechanical_model.get_eet()
def map_grids(shf=np.asarray(models.TM.get_surface_heat_flow(
    format='positive milliwatts')),
        eet=np.asarray(models.MM.get_eet())):
    map_grids = {
        'None': None,
        'Surface Heat Flow': shf,
        'Effective Elastic Thickness': eet
    }
    return map_grids
def cross_section_grids(geotherm=np.asarray(models.TM.get_geotherm()),
        yse_t=np.asarray(models.MM.get_yse()[0]),
        yse_c=np.asarray(models.MM.get_yse()[1])):
    cross_section_grids = {
        'None': None,
        'Geotherm': geotherm,
        'Yield Strength Envelope (T)': yse_t,
        'Yield Strength Envelope (C)': yse_c
    }
        #'Yield Strength Envelope (C)': get_compression_yse(mechanical_model)}
    return cross_section_grids

# Plotly color palettes
def coolwarm_palette():
    coolwarm_cmap = cm.get_cmap('coolwarm')
    coolwarm_cscale = cmap_to_cscale(coolwarm_cmap, 255) 
    return coolwarm_cscale
def afmhot_palette():
    afmhot_cmap = cm.get_cmap('afmhot')
    afmhot_cscale = cmap_to_cscale(afmhot_cmap, 255) 
    return afmhot_cscale
def jet_palette_r():
    jet_white_r_cmap = jet_white_r
    jet_white_r_cscale = cmap_to_cscale(jet_white_r_cmap, 255)
    return jet_white_r_cscale
def jet_palette():
    jet_white_cmap = jet_white
    jet_white_cscale = cmap_to_cscale(jet_white_cmap, 255)
    return jet_white_cscale

# Visualization related methods
def choose_map_colorscale(map_grid): 
    if map_grid == 'Surface Heat Flow':
        color_palette = afmhot_palette()
        color_limits = {'min': None, 'max': None}
    elif map_grid == 'Effective Elastic Thickness':
        color_palette = jet_palette_r()
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
        color_palette = jet_palette_r()
        color_limits = {'min': 0, 'max': 200}
    elif cross_section_grid == 'Yield Strength Envelope (C)':
        color_palette = jet_palette()
        color_limits = {'min': -200, 'max': 0}
    else:
        color_palette = None
        color_limits = {'min': None, 'max': None}
    return {'color_palette': color_palette, 'color_limits': color_limits}
def plot_map_data(latitude, longitude, surface_heat_flow, eet, map_grid,
        show_earthquakes):
    colorscale = choose_map_colorscale(map_grid)
    grid = map_grids(surface_heat_flow, eet)[map_grid]
    if map_grid == 'None':
        marker = {'opacity': 0.2}
        text = None
    else:
        marker = (
            {'color': grid.flatten(),
            'colorscale': colorscale['color_palette'],
            'colorbar': {'title':''},
            'cmin': colorscale['color_limits']['min'],
            'cmax': colorscale['color_limits']['max'],
            'line': {'color': 'rgba(0,0,0,0)'},
            'opacity': 1})
        text = grid.flatten()
    earthquakes_scatter=[]
    if show_earthquakes == ['SE']:
        earthquakes = edf[(
            edf.latitude >= latitude-0.1) & (edf.latitude < (latitude+0.1)
        )]
        earthquakes_scatter = go.Scattergeo(
            lon = earthquakes.longitude,
            lat = earthquakes.latitude,
            #error_y = {'array': earthquakes.depthError},
            #error_x = {'array': earthquakes.horizontalError},
            mode='markers',
            name='earthquakes',
            marker={'color': 'black',
                'size': earthquakes.mag, 'sizemode': 'diameter'},
            hovertext= earthquakes.text,
            hoverinfo='text',
            showlegend=False
         )
    return (
        [go.Scattergeo(
            lon = get_x_grid_2D().flatten(),
            lat = get_y_grid_2D().flatten(),
            text = text,
            marker = marker,
            showlegend = False,
            geo = 'geo'),
        go.Scattergeo(
            lon = np.linspace(-81, -59, 100),
            lat = np.repeat(latitude, 100),
            mode = 'lines',
            hoverinfo = 'skip',
            showlegend = False,
            marker = {'color': 'black'}
            ),
        go.Scattergeo(
            lon = [longitude],
            lat = [latitude],
            marker = {'color': 'blue'},
            hoverinfo = 'skip',
            showlegend = False,
            ),
        earthquakes_scatter
        ]
    )

def plot_cross_section_data(latitude, longitude, geotherm, yse_t, yse_c,
        cross_section_grid, show_earthquakes):
    #index = np.where(get_y_axis() == latitude)[0][0]
    index = np.where(np.isclose(get_y_axis(), latitude))[0][0]
    colorscale = choose_cross_section_colorscale(cross_section_grid)
    grid = cross_section_grids(geotherm, yse_t, yse_c)[cross_section_grid]
    boundaries = get_boundaries()
    layers = get_layers()
    fill='tonexty'
    znan=None
    plots = []
    if cross_section_grid != 'None':
        heatmap = go.Heatmap(
            x = get_x_axis(),
            y = get_z_axis(),
            z = grid[:, index, :].T,
            colorscale = colorscale['color_palette'],
            zmin = colorscale['color_limits']['min'],
            zmax = colorscale['color_limits']['max'],
            zsmooth = 'best',
            #name = name
        )
        znan = grid[:, index, :].T.copy()
        invalid = np.isnan(znan)
        znan[invalid] = 1
        znan[np.invert(invalid)] = None
        fill='none'
        plots.append(heatmap)
    for i, key in enumerate(boundaries.keys()):
        x = get_x_axis()
        y = boundaries[key]['array'].cross_section(latitude=latitude)
        if key == 'topo':
            hoverinfo = 'x'
        else:
            hoverinfo = 'skip'
        if i > 0:
            layer_area = go.Scatter(
                x = x,
                y = y,
                mode='lines',
                hoverinfo='skip',
                name=list(layers)[i-1],
                fill=fill,
                fillcolor=layers[list(layers)[i-1]]['color'],
                marker={'color': layers[list(layers)[i-1]]['color']},
                legendgroup='layer'
            )
            plots.append(layer_area)
        boundary_line = go.Scatter(
            x = x,
            y = y,
            mode='lines',
            hoverinfo=hoverinfo,
            name=key,
            marker={'color': boundaries[key]['color']},
            legendgroup='boundary'
        )
        plots.append(boundary_line)
    if show_earthquakes == ['SE']:
        earthquakes = edf[(
            edf.latitude >= latitude-0.1) & (edf.latitude < (latitude+0.1)
        )]
        earthquakes_scatter = go.Scattergl(
            x = earthquakes.longitude,
            y = - earthquakes.depth,
            #error_y = {'array': earthquakes.depthError},
            #error_x = {'array': earthquakes.horizontalError},
            mode='markers',
            name='earthquakes',
            marker={'color': 'black',
                'size': earthquakes.mag, 'sizemode': 'diameter'},
            hovertext= earthquakes.text,
            hoverinfo='text'
         )
        plots.append(earthquakes_scatter)
    plots.extend([
        go.Scatter(
            x=np.repeat(longitude,100),
            y=np.linspace(10,-180,100),
            showlegend=False,
            hoverinfo='skip',
            mode='lines',
            marker={'color':'black'}
            ),
        go.Heatmap(
            x = get_x_axis(),
            y = get_z_axis(),
            z = znan,
            showscale = False,
            hoverinfo = 'skip'
            )
    ])
    return (
        plots
    )

"""
"""
def plot_yse_chart_data(latitude, longitude, yse_t, yse_c, eet_calc_data, s_max):
    index_lat = np.where(np.isclose(get_y_axis(), latitude))[0][0]
    index_lon = np.where(np.isclose(get_x_axis(), longitude))[0][0]
    share_icd = eet_calc_data['share_icd'][index_lon, index_lat]
    share_moho = eet_calc_data['share_moho'][index_lon, index_lat]
    uc_tuple = eet_calc_data['uc_tuple'][index_lon, index_lat]
    lc_tuple = eet_calc_data['lc_tuple'][index_lon, index_lat]
    lm_tuple = eet_calc_data['lm_tuple'][index_lon, index_lat]
    topo = get_topo()[index_lon, index_lat]
    uc_elastic_top= -uc_tuple[0]+topo
    uc_elastic_bottom = -uc_tuple[1]+topo
    uc_elastic_thickness = uc_tuple[2]
    lc_elastic_top = -lc_tuple[0]+topo
    lc_elastic_bottom =  -lc_tuple[1]+topo
    lc_elastic_thickness = lc_tuple[2]
    lm_elastic_top = -lm_tuple[0]+topo
    lm_elastic_bottom =  -lm_tuple[1]+topo
    lm_elastic_thickness = lm_tuple[2]
    print('uc', uc_elastic_top, uc_elastic_bottom)
    print('lc', lc_elastic_top, lc_elastic_bottom)
    print('lm', lm_elastic_top, lm_elastic_bottom)
    boundaries = get_boundaries()
    layers = get_layers()
    plots=[]
    plots.extend(
        [go.Scattergl(
            x=yse_t[:, index_lat, :][index_lon, :],
            y=get_z_axis(),
            name='yse',
            marker={'color': 'black'},
            fill='tozerox',
            fillcolor='rgb(165,165,165)'
        ),
        go.Scattergl(
            x=yse_c[:, index_lat, :][index_lon, :],
            y=get_z_axis(),
            name='yse',
            marker={'color': 'black'},
            fill='tozerox',
            fillcolor='rgb(165,165,165)',
            showlegend=False
        )]
    )
    for i, key in enumerate(boundaries.keys()):
        x = np.linspace(-1000,1000,100),
        y = np.repeat(boundaries[key]['array'][index_lon, index_lat], 100),
        if i > 0:
            layer_area = go.Scatter(
                x = x[0],
                y = y[0],
                mode='lines',
                hoverinfo='skip',
                name=list(layers)[i-1],
                fill='tonexty',
                fillcolor=layers[list(layers)[i-1]]['color'],
                marker={'color': layers[list(layers)[i-1]]['color']},
                legendgroup='layer'
            )
            plots.append(layer_area)
        boundary_line = go.Scatter(
            x = x[0],
            y = y[0],
            mode='lines',
            hoverinfo='skip',
            name=key,
            marker={'color': boundaries[key]['color']},
            legendgroup='boundary'
        )
        plots.append(boundary_line)
    plots.append(
        go.Scatter(
            x = np.repeat(s_max, 100),
            y = np.linspace(-180,10,100),
            mode='lines',
            name='s_max',
            marker={'color': 'rgb(0,109,134)'})
    )
    return plots
"""
go.Scatter(
    x = np.linspace(-1000,1000,100),
    y = np.repeat(get_slab_lab()[index_lon, index_lat], 100),
    mode='lines',
    name='slab/lab',
    marker=marker_slablab
),
go.Scattergl(
    x = np.repeat(s_max, 2),
    y=[uc_elastic_top, uc_elastic_bottom],
    mode='markers',
    name='uc'
),
go.Scattergl(
    x = np.repeat(s_max, 2),
    y=[lc_elastic_top, lc_elastic_bottom],
    mode='markers',
    name='lc'
),
go.Scattergl(
    x = np.repeat(s_max, 2),
    y=[lm_elastic_top, lm_elastic_bottom],
    mode='markers',
    name='lm'
)
"""
#[go.Scatter(
#x=get_x_axis(),
#y=get_slab_lab().cross_section(latitude=latitude),
#name='yse'
#)
#]


print("After Models Variables M.S.:")
mem()
mem2()

###############################################################################
########################### Dashboard Layout / View ###########################
###############################################################################

app = dash.Dash()

app.scripts.config.serve_locally = True
app.css.config.serve_locally = True

app.css.append_css({"external_url": "https://codepen.io/chriddyp/pen/bWLwgP.css"})

app.config.suppress_callback_exceptions = True

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content'),
    idc.Import()
])

# Pages layouts
def index_layout():
    return html.Div('Index')

def maker_layout():
    return html.Div('Maker')

def explorer_layout():
    return html.Div([
        html.Div(map_layout(),
                 className='three columns',
                 style={'marginLeft': '40px'}),
        html.Div(graphs_layout(),
                 className='six columns',
                 style={'marginLeft': '0'}),
        html.Div(input_layout(), className='three columns'),
        html.Div(json.dumps(t_input), id='thermal_state',
            style={'display':'none'}),
        html.Div(json.dumps(m_input), id='mechanical_state',
            style={'display':'none'}),
        html.Div(jsonpickle.encode({
            'geotherm': np.asarray(models.TM.get_geotherm()),
            'surface_heat_flow': np.asarray(models.TM.get_surface_heat_flow(
                format='positive milliwatts'))}),
            id='thermal_model',
            style={'display':'none'}),
        html.Div(jsonpickle.encode({
            'yse_t': np.asarray(models.MM.get_yse()[0]),
            'yse_c': np.asarray(models.MM.get_yse()[1]),
            'eet': np.asarray(models.MM.get_eet())}),
            id='mechanical_model',
            style={'display':'none'}),
        html.Div(json.dumps({'latitude': -10.}), id='latitude_state',
                 style={'display': 'none'}),
        html.Div(json.dumps({'longitude': -70.}), id='longitude_state',
                 style={'display': 'none'})
    ])

# Sections layouts
def input_layout():
    return [
        html.Div(thermal_input_layout(),
            id='thermal_input',
            style={'border': '1px solid black',
                'height': '500px',
                'margin': '0',
                'margin-top': '30',
                'overflow': 'scroll'}),
        html.Div(mechanical_input_layout(),
            id='mechanical input',
            style={'border': '1px solid black',
                'height': '300px',
                'margin': '0',
                'margin-top': '30'}),
    ]

def thermal_input_layout():
    return html.Div([
            dcc.Checklist(
                id='thermal_options',
                options=[
                    {'label': 'k = f(z)', 'value': 'k_z'},
                    {'label': 'H = f(z)', 'value': 'H_z'},
                    {'label': 'Delta = ICD', 'value': 'delta_icd'},
                    {'label': 't = f(lat)', 'value': 't_lat'}
                ],
                values=[key for key in t_input.keys() if t_input[key] is True]
                ),
            html.Div([
                html.Span('k_uc',style={'display':'inline-block', 'width':'30px'}),
                html.Div(
                    idc.Slider(id='k_uc', min=1., max=5., step=0.1,
                        value=t_input.k_cs),
                    style={'display': 'inline-block', 'width': '300px',
                        'margin-left': 20})
                ]),
            html.Div([
                html.Span('k_lc',style={'display':'inline-block', 'width':'30px'}),
                html.Div(
                    idc.Slider(id='k_lc', min=1., max=5., step=0.1,
                        value=t_input.k_ci),
                    style={'display': 'inline-block', 'width': '300px',
                        'margin-left': 20})
                ]),
            html.Div([
                html.Span('k_lm',style={'display':'inline-block', 'width':'30px'}),
                html.Div(
                    idc.Slider(id='k_lm', min=1., max=5., step=0.1,
                        value=t_input.k_ml),
                    style={'display': 'inline-block', 'width': '300px',
                        'margin-left': 20})
                ]),
            html.Div([
                html.Span('H_uc',style={'display':'inline-block', 'width':'30px'}),
                html.Div(
                    idc.Slider(id='H_uc', min=0., max=5.e-6, step=1.e-7,
                        value=t_input.H_cs),
                    style={'display': 'inline-block', 'width': '300px',
                        'margin-left': 20})
                ]),
            html.Div([
                html.Span('H_lc',style={'display':'inline-block', 'width':'30px'}),
                html.Div(
                    idc.Slider(id='H_lc', min=0., max=5.e-6, step=1.e-7,
                        value=t_input.H_ci),
                    style={'display': 'inline-block', 'width': '300px',
                        'margin-left': 20})
                ]),
            html.Div([
                html.Span('H_lm',style={'display':'inline-block', 'width':'30px'}),
                html.Div(
                    idc.Slider(id='H_lm', min=0, max=5.e-6, step=1.e-7,
                        value=t_input.H_ml),
                    style={'display': 'inline-block', 'width': '300px',
                        'margin-left': 20})
                ]),
            html.Div([
                html.Span('κ',style={'display':'inline-block', 'width':'30px'}),
                html.Div(
                    idc.Slider(id='kappa', min=0., max=3.e-6, step=1.e-7,
                        value=t_input.kappa),
                    style={'display': 'inline-block', 'width': '300px',
                        'margin-left': 20})
                ]),
            html.Div([
                html.Span('Tp',style={'display':'inline-block', 'width':'30px'}),
                html.Div(
                    idc.Slider(id='Tp', min=1000., max=1500., step=50.,
                        value=t_input.Tp),
                    style={'display': 'inline-block', 'width': '300px',
                        'margin-left': 20})
                ]),
            html.Div([
                html.Span('G',style={'display':'inline-block', 'width':'30px'}),
                html.Div(
                    idc.Slider(id='G', min=1.e-4, max=1.e-3, step=1.e-4,
                        value=t_input.G),
                    style={'display': 'inline-block', 'width': '300px',
                        'margin-left': 20})
                ]),
            html.Div([
                html.Span('V',style={'display':'inline-block', 'width':'30px'}),
                html.Div(
                    idc.Slider(id='V', min=1.e4, max=1.e5, step=5.e3,
                        value=t_input.V),
                    style={'display': 'inline-block', 'width': '300px',
                        'margin-left': 20})
                ]),
            html.Div([
                html.Span('b',style={'display':'inline-block', 'width':'30px'}),
                html.Div(
                    idc.Slider(id='b', min=1., max=5., step=1.,
                        value=t_input.b),
                    style={'display': 'inline-block', 'width': '300px',
                        'margin-left': 20})
                ]),
            html.Div([
                html.Span('α',style={'display':'inline-block', 'width':'30px'}),
                html.Div(
                    idc.Slider(id='alpha', min=0., max=45., step=5.,
                        value=t_input.dip),
                    style={'display': 'inline-block', 'width': '300px',
                        'margin-left': 20})
                ]),
            html.Div([
                html.Span('D',style={'display':'inline-block', 'width':'30px'}),
                html.Div(
                    idc.Slider(id='D', min=1.e-3, max=1.e-2, step=1.e-3,
                        value=t_input.D),
                    style={'display': 'inline-block', 'width': '300px',
                        'margin-left': 20})
                ]),
            html.Div([
                html.Span('δ',style={'display':'inline-block', 'width':'30px'}),
                html.Div(
                    idc.Slider(id='delta', min=0., max=30., step=5.,
                        value=t_input.delta),
                    style={'display': 'inline-block', 'width': '300px',
                        'margin-left': 20})
                ]),
            html.Div([
                html.Span('t',style={'display':'inline-block', 'width':'30px'}),
                html.Div(
                    idc.Slider(id='t', min=0., max=50., step=5.,
                        value=t_input.t),
                    style={'display': 'inline-block', 'width': '300px',
                        'margin-left': 20})
                ]),
        ], style={'padding-left':'10'})

def mechanical_input_layout():
    return html.Div([
            html.Div([
                html.Span('Bs_t',style={'display':'inline-block', 'width':'30px'}),
                html.Div(
                    idc.Slider(id='Bs_t', min=0., max=100.e3, step=5.e3,
                        value=m_input.Bs_t),
                    style={'display': 'inline-block', 'width': '300px',
                        'margin-left': 20})
                ]),
            html.Div([
                html.Span('Bs_c',style={'display':'inline-block', 'width':'30px'}),
                html.Div(
                    idc.Slider(id='Bs_c', min=-100.e3, max=0., step=5.e3,
                        value=m_input.Bs_c),
                    style={'display': 'inline-block', 'width': '300px',
                        'margin-left': 20})
                ]),
            html.Div([
                html.Span('e',style={'display':'inline-block', 'width':'30px'}),
                html.Div(
                    idc.Slider(id='e', min=0., max=1.e-14, step=5.e-16,
                        value=m_input.e),
                    style={'display': 'inline-block', 'width': '300px',
                        'margin-left': 20})
                ]),
            html.Div([
                html.Span('R',style={'display':'inline-block', 'width':'30px'}),
                html.Div(
                    idc.Slider(id='R', min=0., max=10., step=1.e-2,
                        value=m_input.R),
                    style={'display': 'inline-block', 'width': '300px',
                        'margin-left': 20})
                ]),
            html.Div([
                html.Span('uc', style={'display':'inline-block', 'width':'30px'}),
                html.Div(
                    dcc.Dropdown(
                        id='uc',
                        options=[{'label':rhe_data[key]['name'],'value':key}
                            for key in rhe_data.keys()],
                        value=m_input.Cs),
                    style={'display':'inline-block', 'width':'300px',
                        'margin-left': 20}
                )
            ]),
            html.Div([
                html.Span('lc', style={'display':'inline-block', 'width':'30px'}),
                html.Div(
                    dcc.Dropdown(
                        id='lc',
                        options=[{'label':rhe_data[key]['name'],'value':key}
                            for key in rhe_data.keys()],
                        value=m_input.Ci),
                    style={'display':'inline-block', 'width':'300px',
                        'margin-left': 20}
                )
            ]),
            html.Div([
                html.Span('lm', style={'display':'inline-block', 'width':'30px'}),
                html.Div(
                    dcc.Dropdown(
                        id='lm',
                        options=[{'label':rhe_data[key]['name'],'value':key}
                            for key in rhe_data.keys()],
                        value=m_input.Ml),
                    style={'display':'inline-block', 'width':'300px',
                        'margin-left': 20}
                )
            ]),
            html.Div([
                html.Span('s_max', style={'display':'inline-block',
                    'width':'30px'}),
                html.Div(
                    idc.Slider(id='s_max', min=0., max=1000., step=10,
                        value=2.e2),
                    style={'display': 'inline-block', 'width': '300px',
                        'margin-left': 20})
                ])
        ], style={'padding-left':'10'})

def map_layout():
    return [
        #
        html.Div([
            dcc.Graph(id='map'),
            dcc.RadioItems(
                id='map_grid_options',
                options=map_grid_options(),
                value=list(map_grid_options()[0].values())[0],
                labelStyle={'display': 'inline-block', 'font-size': '0.8em'}
            )
        ]),
    ]

def graphs_layout():
    return [
        html.Div(cross_section_layout()),
        html.Div(yse_chart_layout())
    ]


def cross_section_layout():
    return [
        dcc.Graph(id='cross_section', style={'margin': '0'}),
        html.Div([
            dcc.RadioItems(
                id='cross_section_grid_options',
                options=cross_section_grid_options(),
                value=list(cross_section_grid_options()[0].values())[0],
                labelStyle={'display': 'inline-block', 'font-size': '0.8em'}
            )
        ], style={'display': 'inline-block', 'margin-left': '50px'}),
        html.Div([
            dcc.Checklist(
                id='show_earthquakes',
                options=[{'label': 'Show earthquakes', 'value': 'SE'}],
                values=['SE'],
                labelStyle={'display': 'inline-block', 'font-size': '0.8em'}
            )
        ], style={'display': 'inline-block', 'margin-left': '20px'})
    ]

def yse_chart_layout():
    return html.Div(dcc.Graph(id='yse', style={'margin': '0', 'margin-top': 40}))

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
def map_graph_layout(latitude, longitude):
    return (
        {'title': 'Map (Lat: {:.1f}, Lon: {:.1f})'.format(latitude,longitude),
        'autosize': False,
        'height': 800,
        'width': 400,
        'margin': {'l':0,'r':0,'t':80,'b':0,'pad':2,'autoexpand':True},
        'legend': {'orientation': 'h'},
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

def cross_section_graph_layout(latitude, longitude):
    return (
        {
        'title': 'Cross Section (Lat: {:.1f}, Lon: {:.1f})'
            .format(latitude, longitude),
        'legend': {'orientation':'h', 'traceorder': 'normal+grouped'},
        'margin': {'t':40,'b':0},
        'xaxis': {'range': [-81.0, -59.0]},
        'yaxis': {'range': [-180.0, 10.0]}
        }
    )

def yse_chart_graph_layout(latitude, longitude):
    return (
        {
        'title': 'Yield Strength Envelope (Lat: {:.1f}, Lon: {:.1f})'
            .format(latitude,longitude),
        'margin': {'t':40},
        'xaxis': {'range': [-1000, 1000.0]},
        'yaxis': {'range': [-180.0, 10.0]},
        'legend': {'traceorder': 'normal+grouped'}
        }
    )

  


print("After Layout M.S.:")
mem()
mem2()

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
    Output('thermal_state', 'children'),
    [Input('thermal_options', 'values'),
     Input('k_uc', 'value'),
     Input('k_lc', 'value'),
     Input('k_lm', 'value'),
     Input('H_uc', 'value'),
     Input('H_lc', 'value'),
     Input('H_lm', 'value'),
     Input('kappa', 'value'),
     Input('Tp', 'value'),
     Input('G', 'value'),
     Input('V', 'value'),
     Input('b', 'value'),
     Input('alpha', 'value'),
     Input('D', 'value'),
     Input('delta', 'value'),
     Input('t', 'value')]
)
def update_thermal_state(thermal_options, k_uc, k_lc, k_lm, H_uc, H_lc, H_lm,
        kappa, Tp, G, V, b, alpha, D, delta, t):
    thermal_state = {
        'k_z': 'k_z' in thermal_options,
        'H_z': 'H_z' in thermal_options,
        'delta_icd': 'delta_icd' in thermal_options,
        't_lat': 't_lat' in thermal_options,
        'k_cs': k_uc,
        'k_ci': k_lc,
        'k_ml': k_lm,
        'H_cs': H_uc,
        'H_ci': H_lc,
        'H_ml': H_lm,
        'kappa': kappa,
        'Tp': Tp,
        'G': G,
        'V': V,
        'b': b,
        'dip': alpha,
        'D': D,
        'delta': delta,
        't': t
    }
    return json.dumps(thermal_state)

@app.callback(
    Output('mechanical_state', 'children'),
    [Input('Bs_t', 'value'),
     Input('Bs_c', 'value'),
     Input('e', 'value'),
     Input('R', 'value'),
     Input('uc', 'value'),
     Input('lc', 'value'),
     Input('lm', 'value'),
     Input('s_max', 'value')]
)
def update_mechanical_state(Bs_t, Bs_c, e, R, uc, lc, lm, s_max):
    mechanical_state = {
        'Bs_t': Bs_t,
        'Bs_c': Bs_c,
        'e': e,
        'R': R,
        'Cs': uc,
        'Ci': lc,
        'Ml': lm,
        's_max': s_max,
    }
    return json.dumps(mechanical_state)

@app.callback(
    Output('thermal_model', 'children'),
    [Input('thermal_state', 'children')]
)
def update_thermal_model(thermal_state):
    print("update_thermal_model called") 
    thermal_state = json.loads(thermal_state)
    thermal_data = DotDict({'t_input': thermal_state, 'trench_age': trench_age})
    thermal_model = compute_tm(thermal_data, models.GM, models.CS)
    geotherm = np.asarray(thermal_model.get_geotherm())
    surface_heat_flow = np.asarray(thermal_model.get_surface_heat_flow(
            format='positive milliwatts'))
    thermal_model = 0
    gc.collect()
    print("After Thermal Model Update M.S.:")
    mem()
    mem2()
    return jsonpickle.encode({'geotherm': geotherm,
        'surface_heat_flow': surface_heat_flow})

@app.callback(
    Output('mechanical_model', 'children'),
    [Input('mechanical_state', 'children'),
     Input('thermal_model', 'children')]
)
def update_mechanical_model(mechanical_state, thermal_model):
    print("update_mechanical_model called") 
    mechanical_state = json.loads(mechanical_state)
    geotherm = jsonpickle.decode(thermal_model)['geotherm']
    print("update_mechanical_model passed") 
    mechanical_data = DotDict({'m_input': mechanical_state,
        'rheologic_data': rhe_data})
    mechanical_model = compute_mm(mechanical_data, models.GM, geotherm,
            models.CS)
    print("After Mechanical Model Update M.S.:")
    mem()
    mem2()
    yse_t = np.asarray(mechanical_model.get_yse()[0])
    yse_c = np.asarray(mechanical_model.get_yse()[1])
    eet = np.asarray(mechanical_model.get_eet())
    eet_calc_data = mechanical_model.get_eet_calc_data()
    s_max = mechanical_model.vars.s_max
    mechanical_model = 0
    #print("After Mechanical Model Pickle M.S.:")
    #mem()
    #mem2()
    return jsonpickle.encode({'yse_t': yse_t, 'yse_c': yse_c, 'eet': eet,
        'eet_calc_data': eet_calc_data, 's_max': s_max})

@app.callback(
    Output('click-data', 'children'),
    [Input('map', 'hoverData')])
def display_click_data(clickData):
    return json.dumps(clickData, indent=2)

@app.callback(
    Output('latitude_state', 'children'),
    [Input('map', 'clickData')])
def update_latitude_state(map_hover_data):
    latitude = map_hover_data['points'][0]['lat']
    latitude_state = {'latitude': latitude}
    return json.dumps(latitude_state)

@app.callback(
    Output('longitude_state', 'children'),
    [Input('cross_section', 'clickData')])
def update_longitude_state(cross_section_hover_data):
    longitude = cross_section_hover_data['points'][0]['x']
    longitude_state = {'longitude': longitude}
    return json.dumps(longitude_state)

@app.callback(
    Output('map', 'figure'),
    [Input('latitude_state', 'children'),
     Input('longitude_state', 'children'),
     Input('thermal_model', 'children'),
     Input('mechanical_model', 'children'),
     #Input('thermal_state', 'children'),
     #Input('mechanical_state', 'children'),
     Input('map_grid_options', 'value'),
     Input('show_earthquakes', 'values')]
)
def update_map(latitude_state, longitude_state, thermal_model, mechanical_model,
        map_grid, show_earthquakes):
    print("update_map called")
    surface_heat_flow = jsonpickle.decode(thermal_model)['surface_heat_flow']
    eet = jsonpickle.decode(mechanical_model)['eet']
    longitude = json.loads(longitude_state)['longitude']
    latitude = json.loads(latitude_state)['latitude']
    data = plot_map_data(latitude, longitude, surface_heat_flow, eet, map_grid,
        show_earthquakes)
    layout = map_graph_layout(latitude, longitude)
    figure = {'data': data, 'layout': layout}
    return figure

@app.callback(
    Output('cross_section', 'figure'),
    [Input('latitude_state', 'children'),
     Input('longitude_state', 'children'),
     Input('thermal_model', 'children'),
     Input('mechanical_model', 'children'),
     #Input('thermal_state', 'children'),
     #Input('mechanical_state', 'children'),
     Input('cross_section_grid_options', 'value'),
     Input('show_earthquakes', 'values')
    ]
)
def update_cross_section(latitude_state, longitude_state, thermal_model,
        mechanical_model, cross_section_grid, show_earthquakes):
    geotherm = jsonpickle.decode(thermal_model)['geotherm']
    yse_t = jsonpickle.decode(mechanical_model)['yse_t']
    yse_c = jsonpickle.decode(mechanical_model)['yse_c']
    print("update_cross_section_called")
    longitude = json.loads(longitude_state)['longitude']
    latitude = json.loads(latitude_state)['latitude']
    data = plot_cross_section_data(latitude, longitude, geotherm, yse_t, yse_c,
            cross_section_grid, show_earthquakes)
    layout = cross_section_graph_layout(latitude, longitude)
    figure = {'data': data, 'layout': layout}
    return figure
@app.callback(
    Output('yse', 'figure'),
    [Input('latitude_state', 'children'),
     Input('longitude_state', 'children'),
     Input('mechanical_model', 'children')]
     #Input('thermal_state', 'children'),
     #Input('mechanical_state', 'children')]
)
def update_yse_chart(latitude_state, longitude_state, mechanical_model):
    mechanical_model = jsonpickle.decode(mechanical_model)
    s_max = mechanical_model['s_max']
    yse_t = mechanical_model['yse_t']
    yse_c = mechanical_model['yse_c']
    eet_calc_data = mechanical_model['eet_calc_data']
    latitude = json.loads(latitude_state)['latitude']
    longitude = json.loads(longitude_state)['longitude']
    data = plot_yse_chart_data(latitude, longitude, yse_t, yse_c,
        eet_calc_data, s_max)
    layout = yse_chart_graph_layout(latitude, longitude)
    figure = {'data': data, 'layout': layout}
    return figure


print("After Controller M.S.:")
mem()
mem2()

###############################################################################

if __name__ == '__main__':
    app.run_server(debug=True)
    pass

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
