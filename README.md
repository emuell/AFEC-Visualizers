# AFEC Visualizers

This repository contains visualization experiments for [AFEC](https://git.github.com/emuell/afec)'s classification models using [plotly dash](https://plotly.com/dash/).


## Explorer

Shows high-level features of an afec database in a simple table. Clicking on an entry shows the sample waveform and plays the audio file.

### Usage

```bash
python explorer.py PATH_TO/afec.db
```

Then open the dash server's URL in your browser. This usually is http://127.0.0.1:8050/

### Dependencies

Python3 with the following pip modules:
-  pysqlite3, plotly, dash, pydub, just_playback


## Classification Cluster

Creates a 2d t-SNE cluster from the afec high-level classification data. Clicking on a point in the plot shows the samples detailed classification scores and the sample waveform and also plays the audio file.

### Usage

```bash
python classification-cluster.py PATH_TO/afec.db
```

Then open the dash server's URL in your browser. This usually is http://127.0.0.1:8050/

### Dependencies

Python3 with the following pip modules:
- pysqlite3, sklearn, densmap-learn, pandas, numpy, plotly, dash, pydub, just_playback
