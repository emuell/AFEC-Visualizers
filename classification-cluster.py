"""
  Show a AFEC classifiers as a 2D t-SNE plot

  Dependencies (pip): pysqlite3, sklearn, densmap-learn, pandas, numpy, plotly, dash, pydub, just_playback
"""

from __future__ import annotations

import sys, os
import sqlite3, json, csv
import pandas as pd
import numpy as np

from pydub import AudioSegment
from just_playback import Playback

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import plotly.express as px

import dash
import dash_core_components as dcc
import dash_html_components as html

from dash.dependencies import Input, Output

# -------------------------------------------------------------------------------------------------

print('Reading AFEC database...')

examples_path = os.path.join(os.path.dirname(sys.argv[0]), 'Examples')
if len(sys.argv) > 1:
  examples_path = sys.argv[1]
dbfile = os.path.join(examples_path, 'afec.db')

conn = sqlite3.connect(dbfile)
conn.text_factory = str 

class_names = json.loads(next(conn.cursor().execute(
    'SELECT classes FROM classes WHERE classifier="Classifiers"'))[0])
category_names = json.loads(next(conn.cursor().execute(
    'SELECT classes FROM classes WHERE classifier="OneShot-Categories"'))[0])

asset_columns = ['filename', 'classes_VS', 'categories_VS', 'class_signature_VR', 'category_signature_VR']
assets_iter = conn.cursor().execute(
    'SELECT ' + ','.join(asset_columns) + ' FROM assets WHERE status="succeeded"')

# -------------------------------------------------------------------------------------------------

print('Extracting classification features to CSV file...')

csv_path = os.path.join(os.path.dirname(dbfile), 'afec-classification.csv')
csv_file = open(csv_path, 'w')
csv_writer = csv.writer(csv_file)
column_names = ['File', 'Class', 'Category', *class_names, *category_names]
csv_writer.writerow(column_names)
for row in assets_iter:
    # see 'asset_columns' above
    filename = row[0] 
    predicted_classes = json.loads(row[1])
    predicted_categories = json.loads(row[2])  
    class_signature = json.loads(row[3]) 
    category_signature = json.loads(row[4]) 
    class_name = predicted_classes[0] if len(predicted_classes) else 'None' 
    category_name = 'Loop' if class_name == 'Loop' else \
      predicted_categories[0] if len(predicted_categories) else 'None'
    csv_writer.writerow(
        [filename, class_name, category_name, *class_signature, *category_signature]
    )
csv_file.close()

# -------------------------------------------------------------------------------------------------

### audio playback status

playback = Playback() 

### plot data

df = pd.read_csv(csv_path)
features = df.loc[:, class_names[0]:category_names[-1]]
    
#### plot color setup

category_colors = {}
category_color_index = 0
category_color_names = [*class_names, *category_names]
for c in category_color_names:
    color_array = px.colors.qualitative.Light24
    category_colors[c] = color_array[category_color_index % len(color_array)]
    category_color_index = category_color_index + 1

# -------------------------------------------------------------------------------------------------

### plot generation

def create_cluster_plot(perplexity, learning_rate):
    print('Applying t-SNE transform...')
    #pca = PCA(n_components=8)
    #projections = pca.fit_transform(features)
    tsne = TSNE(n_components=2, perplexity=perplexity, 
      learning_rate=learning_rate, metric='correlation', init='pca', random_state=244)
    projections = tsne.fit_transform(features)
    print('Creating t-SNE scatter...')
    fig = px.scatter(
        data_frame=projections, 
        x=0, y=1,
        title=examples_path,
        hover_name=df.File,
        color=df.Category,
        color_discrete_map=category_colors,
        category_orders={'color': category_color_names},
    )
    fig.update_layout(clickmode='event')
    fig.update_yaxes(ticklabelposition='inside top', title=None)
    fig.update_xaxes(ticklabelposition='inside top', title=None)
    fig.update_layout(margin={'l':0, 'r':0, 't':40, 'b':0})
    fig.update_traces(marker_size=8)
    return fig

def create_class_strength_figure(filename):
    item = features[df['File'].isin([filename])]
    fig = px.bar(
        title='Class & Category Strenghts',
        data_frame=item, 
        y=item.columns, 
        color_discrete_map=category_colors,
        category_orders={'columns': category_color_names}
    )
    fig.update_yaxes(ticklabelposition='inside top', title=None)
    fig.update_xaxes(ticklabelposition='inside top', title=None)
    fig.update_layout(margin={'l':0, 'r':0, 't':40, 'b':20})
    return fig

def create_waveform_figure(filename):
    abs_filename = filename
    if (not os.path.isabs(filename)):
        abs_filename = os.path.join(os.path.dirname(dbfile), filename)
    print('Playing file...')
    playback.load_file(abs_filename)
    playback.play()
    print('Loading audio file...')
    audiofile = AudioSegment.from_file(abs_filename)
    print('Generating waveform plot...')
    # downsample to avoid too many points in plot - this is slow...
    signal = audiofile.get_array_of_samples()
    signal = np.frombuffer(signal, dtype ='int16')
    downsample_factor = int(len(signal) / 4096) 
    signal = np.interp(np.arange(0, len(signal), downsample_factor), 
      np.arange(0, len(signal)), signal)
    df = pd.DataFrame(data={'Amplitude': signal})
    fig = px.line(
        title=filename,
        data_frame=df,
        x=df.index,
        y=df.Amplitude,
        render_mode='webgl'
    )
    fig.update_yaxes(ticklabelposition='inside top', title=None)
    fig.update_xaxes(ticklabelposition='inside top', title=None)
    fig.update_layout(margin={'l':0, 'r':0, 't':40, 'b':20})
    return fig

# -------------------------------------------------------------------------------------------------

### layout

print('Creating dash...')

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(
    __name__, 
    external_stylesheets=external_stylesheets,
    meta_tags=[
        {
            'name': 'viewport', 
            'content': 'width=device-width, initial-scale=1'
        }
    ]
)

app.config.suppress_callback_exceptions = True

app.layout = html.Div(
    className='background', 
    style={'height': 'calc(100vh - 16px)', 'width': '100%'},
    children=[
        html.Div(
            className='row', 
            style={'height': 'calc(50vh - 8px)', 'width': '100%'},
            children=[
                html.Div(
                    className='twelve columns',
                    style={'height': 'calc(50vh - 48px)', 'width': '100%'},
                    children=[
                        dcc.Loading(
                            id='cluster',
                            type='default',
                            fullscreen=True,
                            children=[dcc.Markdown('Generating Cluster. This may take a while...')]
                        ),
                        html.P(
                            children=[
                                html.Label('Perplexity'),
                                dcc.Slider(
                                    id='tsne-param-slider',
                                    min=5,
                                    max=50,
                                    value=15,
                                    step=None,
                                    marks={5 * i: '{}'.format(5 * i) for i in range(10 + 1)}
                                )
                            ],
                            style={'height': 20, 'padding-right': '8rem', 'display': 'inline-block'}
                        ),
                        html.P(
                            children=[
                                html.Label('Learning Rate'),
                                dcc.Slider(
                                    id='learningrate-param-slider',
                                    min=10,
                                    max=1000,
                                    value=200,
                                    step=None,
                                    marks={200 * i: '{}'.format(200 * i) for i in range(5 + 1)}
                                )
                            ],
                            style={'height': 20, 'padding-right': '8rem', 'display': 'inline-block'}
                        ),
                    ]
                )
            ]
        ),
        html.Div(
            className='row', 
            style={'height': 'calc(50vh - 8px)', 'width': '100%'},
            children=[
                html.Div(id='class-strength', className='six columns', style={'height': 'inherit'}),
                html.Div(id='waveform', className='six columns', style={'height': 'inherit'})
            ]
        ),
    ]
)

### callbacks

@app.callback(
    dash.dependencies.Output('cluster', 'children'),
    [dash.dependencies.Input('tsne-param-slider', 'value'), 
     dash.dependencies.Input('learningrate-param-slider', 'value')])
def update_cluster(slider_value, learning_rate):
    return dcc.Graph(
        id='cluster-graph', 
        figure=create_cluster_plot(slider_value, learning_rate), 
        style={'height': 'calc(50vh - 48px'})

@app.callback(
    dash.dependencies.Output('class-strength', 'children'),
    [dash.dependencies.Input('cluster-graph', 'clickData')])
def update_class_strength(clickData):
    if clickData == None:
        return dcc.Markdown('No sample selected', 
            style={'display': 'flex', 'height': 'inherit', 'alignItems': 'center', 'justifyContent': 'center'})
    hovertext = clickData['points'][0]['hovertext']
    filename = hovertext
    return dcc.Graph(
        id='class-strength-graph', 
        figure=create_class_strength_figure(filename),
        style={'height': 'inherit'})

@app.callback(
    Output('waveform', 'children'),
    Input('cluster-graph', 'clickData'))
def update_waveform(clickData):
    if clickData == None:
        return dcc.Markdown('No sample selected', 
            style={'display': 'flex', 'height': 'inherit', 'alignItems': 'center', 'justifyContent': 'center'})
    hovertext = clickData['points'][0]['hovertext']
    filename = hovertext
    return dcc.Graph(
        id='waveform-graph', 
        figure=create_waveform_figure(filename), 
        style={'height': 'inherit'})

if __name__ == '__main__':
    app.run_server(debug=True)
