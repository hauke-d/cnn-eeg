## DESCRIPTION

This repository contains a CNN-based neural network model intended to classify motor imagery (MI) EEG data. It is developed using the [Physionet MI/ME](https://www.physionet.org/pn4/eegmmidb/) dataset. 

## PREREQISITES

- Physionet data 
	- use the script `eegmmidb/download.sh` (requires wget) **OR**
	- download from https://www.physionet.org/pn4/eegmmidb/
	- adjust the file path in `util.py` to the location of the files in your system
- Python 2 environment with:
	- numpy, scipy, matplotlib
	- tensorflow & keras
	- [pyedflib](http://pyedflib.readthedocs.io/en/latest/) (for data input)


## USAGE

The main.ipynb is an iPython notebook representing the entry point. 
I recommend running it in an Anaconda evironment (which also includes numpy/scipy/matplotlib).
Anaconda can be downloaded from: https://www.anaconda.com/download/

The code represents a starting point for data preparation and training the neural network models.
It does not provide methods for plotting/evaluation/visualization.
