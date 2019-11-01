## DESCRIPTION

This repository contains a CNN-based neural network model intended to classify motor imagery (MI) EEG data. It is developed using the [Physionet MI/ME](https://www.physionet.org/pn4/eegmmidb/) dataset. 

## PREREQUISITES

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

**Note:** The code is no longer maintained and comes without warranty for correctness. It can be freely used, changed, and distributed. If it was helpful to your work, consider citing

> Dose, H., Møller, J. S., Iversen, H. K., & Puthusserypady, S. (2018). An end-to-end deep learning approach to MI-EEG signal classification for BCIs. Expert Systems with Applications, 114, 532–542. https://doi.org/10.1016/j.eswa.2018.08.031
