"""
  Show a AFEC data in a table with audio preview

  Dependencies (pip): pysqlite3, plotly, dash, pydub, just_playback
"""

from __future__ import annotations

import sys, os

import sqlite3, json
import numpy as np
import pandas as pd

from pydub import AudioSegment
from just_playback import Playback

import plotly.express as px

import dash
import dash_core_components as dcc
import dash_html_components as html

from dash_table import DataTable, FormatTemplate
from dash_table.Format import Format, Scheme
from dash.dependencies import Input, Output

# -------------------------------------------------------------------------------------------------

print('Reading AFEC database...')

if len(sys.argv) != 2:
  raise Exception("Missing afec.db path argument")

dbfile = ""
if sys.argv[1].endswith('afec.db'):
  dbfile = os.path.normpath(sys.argv[1])
else:
  dbfile = os.path.normpath(os.path.join(sys.argv[1], 'afec.db'))

connection = sqlite3.connect(dbfile) 
df = pd.read_sql_query('SELECT *, filename as id FROM assets WHERE status="succeeded"', connection, index_col="id")

# --------------------------------------------------------------------------------------------------

print('Converting data for table...')

notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
def number_to_note(number: int) -> tuple:
    if number == -1:
      return ''
    assert 0 <= number <= 127
    octave = int(number / 12)
    note = notes[int(number) % 12]
    if note.endswith('#'):
      return '%s%d' % (note, octave)
    else:
      return '%s-%d' % (note, octave)

df['classes_VS'] = df['classes_VS'].apply(lambda x: ', '.join(json.loads(x)))
df['categories_VS'] = df['categories_VS'].apply(lambda x: ', '.join(json.loads(x)))

df['base_note_R'] = df['base_note_R'].apply(number_to_note)
df['base_note_R'] = np.where(df['base_note_confidence_R'].between(0, 0.25), 
  '-', df['base_note_R']).astype('str')

df['bpm_R'] = df['bpm_R'].apply(lambda x: '{:.2f}'.format(x)).astype('str')
df['bpm_R'] = np.where(df['bpm_confidence_R'].between(0, 0.25), 
  '-', df['bpm_R']).astype('str')

percentageFormat = FormatTemplate.percentage(2)
fixedFormat = Format(precision=2, scheme=Scheme.fixed)

# -------------------------------------------------------------------------------------------------

### audio playback status

playback = Playback() 

# --------------------------------------------------------------------------------------------------

### layout consts

BODY_MARGIN = 8 # assume this is the default html body margin
CONTENT_HEIGHT = 'calc(100vh - {margin}px)'.format(margin = 2*BODY_MARGIN)
TABLE_HEIGHT = 'calc(50vh - {margin}px)'.format(margin = BODY_MARGIN)
FEATURES_HEIGHT = 'calc(50vh - {margin}px)'.format(margin = BODY_MARGIN)

# --------------------------------------------------------------------------------------------------

### figures

def create_data_table():
  return DataTable(
    id='asset-table',
    data=df.to_dict('records'),
    columns=[
      {
        'name': 'Filename',
        'id': 'filename'
      },
      {
        'name': 'Classes',
        'id': 'classes_VS'
      },
      {
        'name': 'Categories',
        'id': 'categories_VS'
      },
      {
        'name': 'Key',
        'id': 'base_note_R'
      },
      {
        'name': 'Conf.',
        'id': 'base_note_confidence_R',
        'type': 'numeric', 
        'format': percentageFormat
      },
      {
        'name': 'Peak dB',
        'id': 'peak_db_R',
        'type': 'numeric', 
        'format': fixedFormat
      },
      {
        'name': 'RMS dB',
        'id': 'rms_db_R',
        'type': 'numeric', 
        'format': fixedFormat
      },
      {
        'name': 'BPM',
        'id': 'bpm_R',
      },
      {
        'name': 'Conf.',
        'id': 'bpm_confidence_R',
        'type': 'numeric', 
        'format': percentageFormat
      },
      {
        'name': 'Brightness',
        'id': 'brightness_R',
        'type': 'numeric', 
        'format': percentageFormat
      },
      {
        'name': 'Noisiness',
        'id': 'noisiness_R',
        'type': 'numeric', 
        'format': percentageFormat
      },
      {
        'name': 'Harmonicity',
        'id': 'harmonicity_R',
        'type': 'numeric', 
        'format': percentageFormat
      }
    ],
    fixed_rows={ 'headers': True, 'data': 0 },
    # TODO: this is wonky: fixed_columns={ 'headers': True, 'data': 1 },
    sort_action='custom',
    sort_mode='single',
    style_table={'height': TABLE_HEIGHT, 'minWidth': '100%', 'overflowX': 'auto', 'overflowY': 'auto'},
    style_cell={ 
      'textOverflow': 'ellipsis', 
      'overflow': 'hidden',
      'width': '100px',
      'maxWidth': '100px',
      'minWidth': '100px',
      'backgroundColor': 'white'
    },
    style_data_conditional=[
        {'if': {'column_id': 'filename'}, 'minWidth': '350px', 'maxWidth': '350px', 'textAlign': 'left', 'direction': 'rtl'},
        {'if': {'column_id': 'file_sample_rate_R'}, 'minWidth': '60px', 'maxWidth': '60px'},
        {'if': {'column_id': 'file_channel_count_R'}, 'minWidth': '60px', 'maxWidth': '60px'},
        {'if': {'column_id': 'file_bit_depth_R'}, 'minWidth': '60px', 'maxWidth': '60px'},
        {'if': {'column_id': 'classes_VS'}, 'minWidth': '120px', 'maxWidth': '120px'},
        {'if': {'column_id': 'categories_VS'}, 'minWidth': '200px', 'maxWidth': '200px'},
    ],
    virtualization=True,
    page_action='none'
)

def create_waveform_figure(filename):
    abs_filename = os.path.normpath(filename)
    if (not os.path.isabs(filename)):
        abs_filename = os.path.join(os.path.dirname(dbfile), abs_filename)
    print('Playing file...')
    playback.load_file(abs_filename)
    playback.play()
    print('Loading audio file...')
    audiofile = AudioSegment.from_file(abs_filename)
    print('Generating waveform plot...')
    signal = audiofile.get_array_of_samples()
    dtype = 'int8'
    if (audiofile.sample_width == 2):
       dtype = 'int16'
    elif (audiofile.sample_width == 4):
       dtype = 'int32'
    signal = np.frombuffer(signal, dtype=dtype)
    # downsample to avoid too many points in plot - this is slow...
    downsample_factor = max(1, int(len(signal) / 32768))
    if downsample_factor > 1:
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
    fig.update_layout(showlegend=False, margin={'l':0, 'r':0, 't':40, 'b':0})
    return fig

# --------------------------------------------------------------------------------------------------

print('Creating dash app...')

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(
  __name__, 
  external_stylesheets=external_stylesheets,
  meta_tags=[
    {
      'name': 'viewport', 
      'content': 'width=device-width, initial-scale=1'
    }
  ])

app.layout = html.Div(
    className='background', 
    style={'height': CONTENT_HEIGHT, 'width': '100%'},
    children=[
      html.Div(
          className='row', 
          style={'height': TABLE_HEIGHT, 'width': '100%'},
          children=[
            create_data_table()
          ]
      ),
      html.Div(
          className='row', 
          style={'height': FEATURES_HEIGHT, 'width': '100%'},
          children=[
            html.Div(
                id='waveform', 
                className='twelve columns', 
                style={'height': 'inherit'}
              ),
              # TODO: add vr-features
              #html.Div(
              #  id='waveform', 
              #  className='six columns', 
              #  style={'height': 'inherit'}
              #),
              #html.Div(
              #  id='vr-features', 
              #  className='six columns', 
              #  style={'height': 'inherit'}
              #)
          ]
      )
    ]
)

# --------------------------------------------------------------------------------------------------

@app.callback(
    Output('asset-table', 'data'),
    Input('asset-table', 'sort_by'))
def sort_table_data(sort_by):
    global df
    if sort_by and len(sort_by):
        dff = df.sort_values(
            sort_by[0]['column_id'],
            ascending=sort_by[0]['direction'] == 'asc',
            inplace=False
        )
    else:
        dff = df
    return dff.to_dict('records')

@app.callback(
    Output('waveform', 'children'),
    Input('asset-table', 'data'),
    Input('asset-table', 'selected_cells'))
def selection_changes(data, selected_cells):
  global df
  if not selected_cells or not len(selected_cells):
    return dcc.Markdown('No sample selected', 
      style={'display': 'flex', 'height': 'inherit', 'alignItems': 'center', 'justifyContent': 'center'})
  row = data[selected_cells[0]['row']]
  filename = row['filename']
  return dcc.Graph(
    id='waveform-graph', 
    style={'height': 'inherit'},
    figure=create_waveform_figure(filename)
  )

if __name__ == '__main__':
    app.run_server(debug=True)
