# MCCA

This repository contains the code from the paper [Inter-individual single-trial classification of MEG data using M-CCA](https://doi.org/10.1016/j.neuroimage.2023.120079). The data to reproduce results from the paper is available upon request.

## Installation

Use venv or conda to create a new virtual environment. The code was only tested with python3.8 and the exact dependencies listed in requirements.txt. Example installation on Unix/macOS:

```
python -m venv mcca_env
source mcca_env/bin/activate
pip install -r requirements.txt
```

## Usage

### As a library

The main class that computes the MCCA space and can transform data between sensor, PCA, and MCCA space is in MCCA.py.
MCCA_transformer.py implements MCCA transformation for use in sklearn pipelines and includes methods to include new subjects into an already fitted MCCA space (fit_online and transform_online).

## Running the analysis from the paper

Specify data and results directories and MCCA parameters in config.yaml, then run run.py. The data is not included in this repo and available upon request.
