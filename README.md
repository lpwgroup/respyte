respyte
==============================
[//]: # (Badges)
[![Travis Build Status](https://travis-ci.org/REPLACE_WITH_OWNER_ACCOUNT/respyte.png)](https://travis-ci.org/REPLACE_WITH_OWNER_ACCOUNT/respyte)
[![AppVeyor Build status](https://ci.appveyor.com/api/projects/status/REPLACE_WITH_APPVEYOR_LINK/branch/master?svg=true)](https://ci.appveyor.com/project/REPLACE_WITH_OWNER_ACCOUNT/respyte/branch/master)
[![codecov](https://codecov.io/gh/REPLACE_WITH_OWNER_ACCOUNT/respyte/branch/master/graph/badge.svg)](https://codecov.io/gh/REPLACE_WITH_OWNER_ACCOUNT/respyte/branch/master)

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
To calculate espf data, run this command from esp folder:
`python esp_generator.py`

After the calculation is done, copy 'molecules' folder in 'input' to 'input' in resp folder and with an appropriate respyte.yml file, run this command:
`python resp_optimizer.py`


#### Copyright

Copyright (c) 2018, Hyesu Jang


#### Acknowledgements

Project based on the
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms)

