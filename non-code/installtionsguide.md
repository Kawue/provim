# Getting started

This Readme gives a quick introduction into the installation and execution of the workflow for msi-preprocessing.

## Installation

Use [Anaconda](https://www.anaconda.com/distribution/) to create a python environment from the environment.yml (recommended)

```bash
conda env create -f environment.yml
```

## Usage
After the environment is installed, you have to activate it.
```bash 
conda activate msi-preprocessing
```
To get the workflow started, you can either use the msi-preprocessing.sh or start the programs yourself.

```bash
sh msi-preprocessing.sh
python <path-to-program>/<programname> -h
```

It is possible to start some programs (matrix\_preprocessing.py, matrix\_postprocessing.py, workflow\_peakpicking.py) with multiple files (or directories) by listing them with the fileparameter seperated by spaces.
