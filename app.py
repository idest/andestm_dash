############################## Memory State Test ##############################
import resource

def mem():
    print('### Memory usage         : % 2.2f MB ###' % round(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.0,1)
    )
###############################################################################

print("Initial M.S.:")
mem()

# Python libs
from textwrap import dedent as d
import json

# Dash libs
import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
import idest_dash_components as idc
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
from compute import compute
from utils import DotDict, cmap_to_cscale
from meccolormap import jet_white_r

print("After My Modules Imports M.S.:")
mem()

print("After All Module Imports M.S.:")
mem()

###############################################################################
########################## Data Manipulation / Model ##########################
###############################################################################

# Static Input
gm_data = np.loadtxt('data/Modelo.dat')
areas = np.loadtxt('data/areas.dat')
trench_age = np.loadtxt('data/PuntosFosaEdad.dat')
rhe_data = setup.read_rheo('data/Rhe_Param.dat')
print(rhe_data)
print(rhe_data.keys())

# User Input
t_input = setup.readVars('VarTermal.txt')
m_input = setup.readVars('VarMecanico.txt')

print("After Input M.S.:")
mem()

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
def get_geometry():
    return models.GM.get_3D_geometric_model()
def get_geotherm():
    return models.TM.get_geotherm()
def get_tension_yse():
    return models.MM.get_yse()[0]
def get_compression_yse():
    return models.MM.get_yse()[1]
def get_surface_heat_flow():
    return models.TM.get_surface_heat_flow()
def get_effective_elastic_thickness():
    return models.MM.get_eet()
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
def plot_map_data(latitude, longitude, map_grid):
    colorscale = choose_map_colorscale(map_grid)
    grid = map_grids()[map_grid]
    if map_grid == 'None':
        marker = {'opacity': 0.2}
    else:
        marker = (
            {'color': grid.flatten(),
            'colorscale': colorscale['color_palette'],
            'colorbar': {'title':''},
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
            )
        ]
    )

def plot_cross_section_data(latitude, longitude, cross_section_grid):
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
            hoverinfo='skip'
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
            hoverinfo='x',
            name='topo'),
        go.Scatter(
            x=get_x_axis(),
            y=get_icd().cross_section(latitude=latitude),
            hoverinfo='skip',
            name='icd'),
        go.Scatter(
            x=get_x_axis(),
            y=get_moho().cross_section(latitude=latitude),
            hoverinfo='skip',
            name='moho'),
        go.Scatter(
            x=get_x_axis(),
            y=get_slab_lab().cross_section(latitude=latitude),
            hoverinfo='skip',
            name='slab/lab'),
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
            )]
    )

def plot_yse_chart_data(latitude, longitude):
    yse = get_tension_yse()
    print(get_topo().shape)
    print(yse.cross_section(latitude=latitude).shape)
    print(len(yse.cross_section(latitude=latitude).cross_section(longitude=longitude)))
    print(len(get_z_axis()))
    return (
            #
            [ go.Scattergl(
                x=yse.cross_section(latitude=latitude).cross_section(longitude=longitude),
                y=get_z_axis(),
                name='yse'
                )]
            #[go.Scatter(
            #x=get_x_axis(),
            #y=get_slab_lab().cross_section(latitude=latitude),
            #name='yse'
            #)
            #]
            )


print("After Models Variables M.S.:")
mem()

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
                options=[
                    {'label': 'k = f(z)', 'value': 'k_z'},
                    {'label': 'H = f(z)', 'value': 'H_z'},
                    {'label': 'Delta = ICD', 'value': 'delta_icd'},
                    {'label': 't = f(lat)', 'value': 't_lat'}
                ],
                values=['t_lat']
                ),
            html.Div([
                html.Span('k_uc',style={'display':'inline-block', 'width':'30px'}),
                html.Div(
                    idc.Slider(min=1., max=5., step=0.1, value=3.),
                    style={'display': 'inline-block', 'width': '300px',
                        'margin-left': 20})
                ]),
            html.Div([
                html.Span('k_lc',style={'display':'inline-block', 'width':'30px'}),
                html.Div(
                    idc.Slider(min=1., max=5., step=0.1, value=3.),
                    style={'display': 'inline-block', 'width': '300px',
                        'margin-left': 20})
                ]),
            html.Div([
                html.Span('k_lm',style={'display':'inline-block', 'width':'30px'}),
                html.Div(
                    idc.Slider(min=1., max=5., step=0.1, value=3.),
                    style={'display': 'inline-block', 'width': '300px',
                        'margin-left': 20})
                ]),
            html.Div([
                html.Span('H_uc',style={'display':'inline-block', 'width':'30px'}),
                html.Div(
                    idc.Slider(min=0., max=5.e-6, step=1.e-7, value=3.e-6),
                    style={'display': 'inline-block', 'width': '300px',
                        'margin-left': 20})
                ]),
            html.Div([
                html.Span('H_lc',style={'display':'inline-block', 'width':'30px'}),
                html.Div(
                    idc.Slider(min=0., max=5.e-6, step=1.e-7, value=3.e-6),
                    style={'display': 'inline-block', 'width': '300px',
                        'margin-left': 20})
                ]),
            html.Div([
                html.Span('H_lm',style={'display':'inline-block', 'width':'30px'}),
                html.Div(
                    idc.Slider(min=1.e-6, max=5.e-6, step=1.e-7, value=3.e-6),
                    style={'display': 'inline-block', 'width': '300px',
                        'margin-left': 20})
                ]),
            html.Div([
                html.Span('κ',style={'display':'inline-block', 'width':'30px'}),
                html.Div(
                    idc.Slider(min=0., max=3.e-6, step=1.e-7, value=1.e-6),
                    style={'display': 'inline-block', 'width': '300px',
                        'margin-left': 20})
                ]),
            html.Div([
                html.Span('Tp',style={'display':'inline-block', 'width':'30px'}),
                html.Div(
                    idc.Slider(min=1000., max=1500., step=50., value=1250.),
                    style={'display': 'inline-block', 'width': '300px',
                        'margin-left': 20})
                ]),
            html.Div([
                html.Span('G',style={'display':'inline-block', 'width':'30px'}),
                html.Div(
                    idc.Slider(min=1.e-4, max=1.e-3, step=1.e-4, value=4.e-4),
                    style={'display': 'inline-block', 'width': '300px',
                        'margin-left': 20})
                ]),
            html.Div([
                html.Span('V',style={'display':'inline-block', 'width':'30px'}),
                html.Div(
                    idc.Slider(min=1.e4, max=1.e5, step=5.e3, value=6.5e4),
                    style={'display': 'inline-block', 'width': '300px',
                        'margin-left': 20})
                ]),
            html.Div([
                html.Span('b',style={'display':'inline-block', 'width':'30px'}),
                html.Div(
                    idc.Slider(min=1., max=5., step=1., value=1.),
                    style={'display': 'inline-block', 'width': '300px',
                        'margin-left': 20})
                ]),
            html.Div([
                html.Span('α',style={'display':'inline-block', 'width':'30px'}),
                html.Div(
                    idc.Slider(min=0., max=45., step=5., value=20.),
                    style={'display': 'inline-block', 'width': '300px',
                        'margin-left': 20})
                ]),
            html.Div([
                html.Span('D',style={'display':'inline-block', 'width':'30px'}),
                html.Div(
                    idc.Slider(min=1.e-3, max=1.e-2, step=1.e-3, value=1.e-3),
                    style={'display': 'inline-block', 'width': '300px',
                        'margin-left': 20})
                ]),
            html.Div([
                html.Span('δ',style={'display':'inline-block', 'width':'30px'}),
                html.Div(
                    idc.Slider(min=0., max=30., step=5., value=10.),
                    style={'display': 'inline-block', 'width': '300px',
                        'margin-left': 20})
                ]),
            html.Div([
                html.Span('t',style={'display':'inline-block', 'width':'30px'}),
                html.Div(
                    idc.Slider(min=0., max=50., step=5., value=30.),
                    style={'display': 'inline-block', 'width': '300px',
                        'margin-left': 20})
                ]),
        ], style={'padding-left':'10'})

def mechanical_input_layout():
    return html.Div([
            html.Div([
                html.Span('Bs_t',style={'display':'inline-block', 'width':'30px'}),
                html.Div(
                    idc.Slider(min=0., max=100.e3, step=5.e3, value=20.e3),
                    style={'display': 'inline-block', 'width': '300px',
                        'margin-left': 20})
                ]),
            html.Div([
                html.Span('Bs_c',style={'display':'inline-block', 'width':'30px'}),
                html.Div(
                    idc.Slider(min=-100.e3, max=0., step=5.e3, value=-55.e3),
                    style={'display': 'inline-block', 'width': '300px',
                        'margin-left': 20})
                ]),
            html.Div([
                html.Span('e',style={'display':'inline-block', 'width':'30px'}),
                html.Div(
                    idc.Slider(min=0., max=1.e-14, step=5.e-16, value=1.e-15),
                    style={'display': 'inline-block', 'width': '300px',
                        'margin-left': 20})
                ]),
            html.Div([
                html.Span('R',style={'display':'inline-block', 'width':'30px'}),
                html.Div(
                    idc.Slider(min=0., max=10., step=1.e-2, value=8.31),
                    style={'display': 'inline-block', 'width': '300px',
                        'margin-left': 20})
                ]),
            html.Div([
                html.Span('uc', style={'display':'inline-block', 'width':'30px'}),
                html.Div(
                    dcc.Dropdown(
                        options=[{'label':rhe_data[key]['name'],'value':key}
                            for key in rhe_data.keys()],
                        value='9'),
                    style={'display':'inline-block', 'width':'300px',
                        'margin-left': 20}
                )
            ]),
            html.Div([
                html.Span('lc', style={'display':'inline-block', 'width':'30px'}),
                html.Div(
                    dcc.Dropdown(
                        options=[{'label':rhe_data[key]['name'],'value':key}
                            for key in rhe_data.keys()],
                        value='28'),
                    style={'display':'inline-block', 'width':'300px',
                        'margin-left': 20}
                )
            ]),
            html.Div([
                html.Span('lm', style={'display':'inline-block', 'width':'30px'}),
                html.Div(
                    dcc.Dropdown(
                        options=[{'label':rhe_data[key]['name'],'value':key}
                            for key in rhe_data.keys()],
                        value='22'),
                    style={'display':'inline-block', 'width':'300px',
                        'margin-left': 20}
                )
            ]),
            html.Div([
                html.Span('s_max',style={'display':'inline-block', 'width':'30px'}),
                html.Div(
                    idc.Slider(min=0., max=1000., step=1.e2, value=2.e2),
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
        ], className='twelve columns'),
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
        'legend': {'orientation':'h'},
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
        'margin': {'t':40}
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
    [Input('latitude_state', 'children'),
     Input('longitude_state', 'children'),
     Input('map_grid_options', 'value')]
)
def update_map(latitude_state, longitude_state, map_grid):
    print("update_map called")
    longitude = json.loads(longitude_state)['longitude']
    latitude = json.loads(latitude_state)['latitude']
    data = plot_map_data(latitude, longitude, map_grid)
    layout = map_graph_layout(latitude, longitude)
    figure = {'data': data, 'layout': layout}
    return figure

@app.callback(
    Output('cross_section', 'figure'),
    [Input('latitude_state', 'children'),
     Input('longitude_state', 'children'),
     Input('cross_section_grid_options', 'value')
    ]
)
def update_cross_section(latitude_state, longitude_state, cross_section_grid):
    print("update_cross_section_called")
    longitude = json.loads(longitude_state)['longitude']
    latitude = json.loads(latitude_state)['latitude']
    data = plot_cross_section_data(latitude, longitude, cross_section_grid)
    layout = cross_section_graph_layout(latitude, longitude)
    figure = {'data': data, 'layout': layout}
    return figure
@app.callback(
    Output('yse', 'figure'),
    [Input('latitude_state', 'children'), Input('longitude_state', 'children')]
)
def update_yse_chart(latitude_state, longitude_state):
    latitude = json.loads(latitude_state)['latitude']
    longitude = json.loads(longitude_state)['longitude']
    data = plot_yse_chart_data(latitude, longitude)
    layout = yse_chart_graph_layout(latitude, longitude)
    figure = {'data': data, 'layout': layout}
    return figure
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
