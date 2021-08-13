# AFEC Visualizers

This repository contains vizualization experiments for [AFEC](https://git.github.com/emuell/afec)'s classification models using [plotly dash](https://plotly.com/dash/).


## Explorer

Shows high level features of an afec database in a table and shows the sample waveform. Clicking on an entry plays the audio file.

### Usage

```bash
python explorer.py PATH_TO/afec.db
```

Then open the dash server's URL in your browser. This usually is http://127.0.0.1:8050/

### Dependencies

Python3 with the following pip modules:
-  pysqlite3, plotly, dash, pydub, just_playback


## Classification Cluster

Creates a 2d T-SNE cluster from the afec high level classification data. Clicking on a point in the plot shows the samples detailed classification scores and the sample waveform. Clicking on an entry plays also plays the audio file.

### Usage

```bash
python classification-cluster.py PATH_TO/afec.db
```

Then open the dash server's URL in your browser. This usually is http://127.0.0.1:8050/

### Dependencies

Python3 with the following pip modules:
- pysqlite3, sklearn, densmap-learn, pandas, numpy, plotly, dash, pydub, just_playback
