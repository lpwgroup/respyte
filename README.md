respyte
==============================
[//]: # (Badges)
[![Build Status](https://travis-ci.org/lpwgroup/respyte.svg?branch=master)](https://travis-ci.org/lpwgroup/respyte)
[![Codecov coverage](https://img.shields.io/codecov/c/github/lpwgroup/respyte.svg?logo=Codecov&logoColor=white)](https://codecov.io/gh/lpwgroup/respyte) 

implementation of open-source version of RESP method

## Installation
From a clean Miniconda installation, run the following:

```
conda create --name respyte python=3.6
source activate respyte
conda install -c psi4 psi4
conda install -c openeye openeye-toolkits
conda install pandas scipy pyyaml matplotlib future
conda install -c conda-forge pymbar
conda install -c omnia forcebalance
```

## Running
Before running 'esp_generator.py', Please navigate 'data' folder which contains sample input folder. This will give you 
some idea about what kind of data structure 'esp_generator.py' and 'resp_optimizer.py' can handle. 

First, to calculate electrostatic potential (and electric field) using Psi4, create input folder with appropriate data structure and run this command inside the directory where your 'input' exists:

`python esp_generator.py`

If the calculation is successfully done, you can find files with .espf extension in each subfolder,'input/molecules/mol(i)/conf(j)'. This stores all the grid point informations with specific file format for 'resp_optimizer.py'. Then you are ready to go to the next step. Write respyte.yml inside 'input/' and run this command:

`python resp_optimizer.py`


#### Copyright

Copyright (c) 2018, Hyesu Jang


#### Acknowledgements

Project based on the
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms)

