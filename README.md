# MCCA

This repository reproduces the results from the paper "Inter-individual single-trial classification of MEG data using M-CCA". Results reported in the paper were computed without setting the random state of solvers. This has been fixed by setting the random seed for solvers to 0 in this repo. Consequently, results from this repo may be slightly different (at most +/- 0.01 balanced accuracy) than results reported in the paper. The final version of the paper will have its numbers and figures updated to exactly match the results in this repo.

## Folder structure

```
./data/                                 Contains single-trial data from the MEG experiment as numpy arrays (no metadata),
                                        both with and without temporal signal space separation (tSSS) realignment. 
./results/                              Results from running the analysis are stored in the corresponding subfolders: 
./results/intra_subject_decoder         Intra-subject decoding results which are used as a baseline 
./results/inter_subject_decoder         Inter-subject decoding results in sensor, tSSS and MCCA space 
./results/inter_subject_decoder/online  Simulated online decoding results 
./results/permutation_test              Permutation test results
```

## Installation

Use venv or conda to create a new virtual environment. The code was only tested with python3.8 and the exact dependencies listed in requirements.txt. Example installation on Unix/macOS:

```
python3.8 -m venv mcca_env
source mcca_env/bin/activate
pip install -r requirements.txt
```

## Running the analysis

```
python3.8 run.py
```

Warning: running simulated online decoding and permutation testing takes a long time (days) to complete. Comment them out or reduce number of permutations  in run.py to improve running time.