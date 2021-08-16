# AFEC Visualizers

This repository contains visualization experiments for [AFEC](https://git.github.com/emuell/afec)'s classification models using [plotly dash](https://plotly.com/dash/).


## Explorer

Shows high-level features of an afec database in a simple table. Clicking on an entry shows the sample waveform and plays the audio file.

![AFEC-Explorer](https://user-images.githubusercontent.com/11521600/129560012-d2cdc0a3-a4b0-4747-a00a-5722539a5f1b.PNG)

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

![AFEC-Cluster](https://user-images.githubusercontent.com/11521600/129560039-7194c36d-7a39-47b7-9b27-ea49dd2cab20.PNG)

### Usage

```bash
python classification-cluster.py PATH_TO/afec.db
```

Then open the dash server's URL in your browser. This usually is http://127.0.0.1:8050/

### Dependencies

Python3 with the following pip modules:
- pysqlite3, sklearn, densmap-learn, pandas, numpy, plotly, dash, pydub, just_playback
