# respyte
implementation of open-source version of RESP method

# Installation
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

# Running
To run the code, run this command from the root folder:
`python resp_optimizer.py`

