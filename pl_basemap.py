

m = Basemap()

# Make trace-generating function (return a Scatter object)
def make_scatter(x,y):
    return go.Scatter(
        x=x,
        y=y,
        mode='lines',
        line=go.Line(color="black"),
        name=' '  # no name on hover
    )

# Functions converting coastline/country polygons to lon/lat traces
def polygons_to_traces(poly_paths, N_poly):
    '''
    pos arg 1. (poly_paths): paths to polygons
    pos arg 2. (N_poly): number of polygon to convert
    '''
    traces = []  # init. plotting list

    for i_poly in range(N_poly):
        poly_path = poly_paths[i_poly]

        # get the Basemap coordinates of each segment
        coords_cc = np.array(
            [(vertex[0],vertex[1])
             for (vertex,code) in poly_path.iter_segments(simplify=False)]
        )

        # convert coordinates to lon/lat by 'inverting' the Basemap projection
        lon_cc, lat_cc = m(coords_cc[:,0],coords_cc[:,1], inverse=True)

        # add plot.ly plotting options
        traces.append(make_scatter(lon_cc,lat_cc))

    return traces

# Function generating coastline lon/lat traces
def get_coastline_traces():
    poly_paths = m.drawcoastlines().get_paths() # coastline polygon paths
    N_poly = 91  # use only the 91st biggest coastlines (i.e. no rivers)
    return polygons_to_traces(poly_paths, N_poly)

# Function generating country lon/lat traces
def get_country_traces():
    poly_paths = m.drawcountries().get_paths() # country polygon paths
    N_poly = len(poly_paths)  # use all countries
    return polygons_to_traces(poly_paths, N_poly)

# Get list of of coastline and country lon/lat traces
traces_cc = get_coastline_traces()+get_country_traces()
trace1 = go.Heatmap(
    z = eet.T,
    x = GM.cs.get_axes()[0],
    y = GM.cs.get_axes()[1],
    zmin=0,
    zmax=100,
    colorscale=jet
)
data = go.Data([trace1]+traces_cc)
axis_style = dict(
    zeroline=False,
    showline=False,
    showgrid=True,
    #ticks='',
    #showticklabels=False
)
layout = go.Layout(
    showlegend=False,
    hovermode='closest',
    xaxis = go.XAxis(
        axis_style,
        range=[GM.cs.get_axes()[0][0], GM.cs.get_axes()[0][-1]]
    ),
    yaxis = go.YAxis(
        axis_style,
        range=[GM.cs.get_axes()[1][-1], GM.cs.get_axes()[1][0]],
        scaleanchor='x'
    ),
)


html.Div([
    dcc.Graph(
        id='test',
        figure = {
            'data': data,
            'layout': layout
        }
    )
], className="twelve columns")
