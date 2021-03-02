
***respyte***
==============================
[//]: # (Badges)
[![Build Status](https://travis-ci.org/lpwgroup/respyte.svg?branch=re-formatting)](https://travis-ci.org/lpwgroup/respyte)
[![codecov](https://codecov.io/gh/lpwgroup/respyte/branch/re-formatting/graph/badge.svg)](https://codecov.io/gh/lpwgroup/respyte)

A python implementation of open-source version of RESP method

## Short description of respyte beta ver.
The main goals of the reformatting are (1) implementation of better user interface and (2) to support various charge models (fuzzy charge model, virtual site etc.)

Newly added features(updated 2021-03-01): (1) normalization and (2) the usage of electric field term in fitting.

### 1. conda environment and dependent packages 
Create a new environment and install all dependencies
```
conda create -n respyte-beta python=3.7
source activate respyte-beta
conda install -c psi4 psi4 
conda install scipy networkx pyyaml sympy
conda install -c rdkit rdkit 
```
Optionally, one can write fitting results into `.mol2` files if OpenEye toolkit can be imported(only support point charge model).
```
conda install -c openeye openeye-toolkits
```
Clone the repository and checkout to re-formatting branch (Once it is merged to master, no longer need to switch to re-formatting branch.)
```
git clone https://github.com/lpwgroup/respyte.git
git checkout re-formatting
```
And run `setup.py` for installation of the main package:
```
python setup.py install
```

### 2. Command-line interface 
Before running, Please navigate `/data/input.sample` which contains sample of input folder and files. This will give you some idea on the data structure the package can handle.

#### 2.1. ESP calculation using Psi4
First, to calculate electrostatic potential (and electric field) using Psi4, create input folder with an appropriate data structure and an input file,  `input.yml`. Then run this command:
```
respyte-esp_generator input-dir-name
```
#### 2.2. Charge fitting 

Once the ESP calculation is successfully done, you are ready to go to the next step. You can find the  output  files with `.espf` extension in each subfolder,`input-dir-name/molecules/molecule-name/conf(j)`,  which  store all the grid point informations with a file format that the respyte package can understand.

Write `respyte.yml` inside `input-dir-name` and run this command:

```
respyte-optimizer input-dir-name
```

### 3. Using respyte as a python module

example of the use of the package as a module can be found in `/data/ipynb_example/example.ipynb`. 
    
To  use *respyte* module, there are three main components you may want to know:

#### (1) `molecules.py`
: Creates **molecule object**, which takes coordinate file and espf file and store data for charge fitting. The way to generate molecule object is: 
```
# generate molecule object
molecule = respyte_molecule(molecule_name= 'meoh', coord_fnm='test.pdb', espf_fnm='test.espf')
# set a net charge of the molecule
molecule.set_net_charge(0)
```
main attributes of molecule object: 
- `self.mol` : forcebalance molecule object. storing coordinates, element, and vdw radius of all atoms 
- `self.polar_atom_indices` : list of indice of polar atoms (all atoms except for alkyl carbons and alkyl hydrogens)
- `self.atom_equiv` : a dictionary, whose keys are equivalence level('nosym', 'connectivity', 'relaxed_connectivity', 'symbol', 'symbol2') and the values are dictionaries, having 'equivs', a list of equiv values for the corresponding equivalence level, and 'info', a dictionary contains definition of each equiv value. Note that term, 'equiv' is the replacement of 'atomid'
- For  example, atom_equiv for methanol molecule are: 
```
In [4]: molecule.atom_equiv.keys()
Out[4]: dict_keys(['nosym', 'connectivity', 'relaxed_connectivity', 'symbol', 'symbol2'])
```

```
In [5]: molecule.atom_equiv['connectivity']['equivs']
Out[5]: [2, 0, 3, 1, 1, 1]
```
Atom number 4 to 6 are assigned to the same equiv value 1 in the equivalence level 'connectivity', meaning that the atoms are equivalent in the level. (they are three alkyl hydrogens in a methanol molecule.)

And the information of the equiv value 2 can be searched like this:
```
In [6]: molecule.atom_equiv['connectivity']['info'][2]
Out[6]: 
[{'molname': 'meoh',
  'resname': '<0>',
  'atomname': 'O1',
  'elem': 'O',
  'vdw_radius': 1.52}]
```
- `self.gridxyz`, `self.espval`, `self.efval` : information about grids and QM values
- All the units of length are Angstrom.
- And **molecules object**  take multiple molecule objects and reassign atomids so that there’s no atom id used multiple times for different atoms. 
```
# create a respyte molecules object
molecules = respyte_molecules()
# add onle molecule object to molecules object
molecules.add_molecule(molecule) 
```
#### (2) `model.py`
Model object takes molecules object and build a parameter set, `parms`, which is a list of initial guesses of parameters to be fitted to QM properties. Currently point charge model and fuzzy charge model are implemented. 

#### (3) `objective.py`

: **Objective object** (1) take a set of parameters to be fitted and fixed parameter information from model object (2) combine target objects and penalty function to build total objective function of the system (3) calculate the objective function, gradient and hessian of the objective function at a given point inside optimizer, and (4) keep the fitted parameters the information of each parameter into `parms` and  `parm_info`. 
An example of how to build objective object:
```
# create an objective from molecules
objective = respyte_objective(molecules)

# add model and build parameter set to be fitted
parameter_types ={'charge':'connectivity'}
objective.add_model(model_type='point_charge', parameter_types=parameter_types)

# add targets 
targets = [{'type': 'esp',  'weight': 1.0}] 
objective.add_target(targets)

# add penalty function for parameter regularization 
penalty = {'ptype':'L1', 'a':0.001, 'b':0.1}
objective.add_penalty(penalty)
```
And the created parameter set will be like this. (note that the values are initial guesses.)
```
In []: objective.parms
Out[]: [0, 0, 0, 0, 1]

In []: objective.parm_info
Out[]: # each element means: equiv value, parameter type, equivalence level of the parameter.
[[0, 'charge', 'connectivity'],
 [1, 'charge', 'connectivity'],
 [2, 'charge', 'connectivity'],
 [3, 'charge', 'connectivity'],
 ['l0', 'lambda', 'connectivity']] 
```

#### (4) `optimizer.py`
: **Optimizer object** take objective object and run Newton-Rapson method using scipy and return the values once the convergence criterion is  met. Once the objective object is generated, you can run the optimization like this: 
```
# define an optimizer and run it.
optimizer = respyte_optimizer(objective)
optimizer.run(verbose=True)
```

#### (5) `procedure.py`
It defines a main function called `resp` which runs single-stage or two-stage fitting  procedure. 

- Example 1. Analytic solution of single-stg-fit to ESP
```
parameter_types = {'charge': 'connectivity'}
model_type='point_charge'
penalty =  {'ptype': 'L1', 'a': 0.001, 'b': 0.1}
# default target is [{'type': 'esp',  'weight': 1.0}], which specifies ESP-only fitting
resp(molecules, model_type, parameter_types, penalty=penalty, procedure=1)
```
- Example 2. Analytic solution of single-stg-fit to **ESP and EF**
```
model_type='point_charge'
parameter_types = {'charge': 'connectivity'}
penalty =  {'ptype': 'L1', 'a': 0.001, 'b': 0.1}
targets = [{'type': 'esp',  'weight': 0.5}, {'type': 'ef',  'weight': 0.5}] 
resp(molecules, model_type, parameter_types, targets=targets, penalty=penalty, procedure=1)
```
- Example 3. Analytic solution  of two-stg-fit to ESP
```
model_type = 'point_charge'
parameter_types = {'charge': 'connectivity'}
penalty =  {'type': 'L1', 'a': 0.001, 'b': 0.1}
resp(molecules, model_type, parameter_types, penalty=penalty, procedure=2)
```
- Example 4. **Fuzzy charge model**(numerical  solution of single-stg-fit to ESP) 
```
model_type = 'fuzzy_charge'
parameter_types = {'charge': 'connectivity', 'alpha': 'symbol'}
q_core_type = 'n_outer_elecs'
alpha0 = 3
penalty =  {'type': 'L1', 'a': 0.001, 'b': 0.1, 'c':0.1}
resp(molecules, model_type, parameter_types, q_core_type=q_core_type, alpha0=alpha0, penalty=penalty, procedure=2)
```

#### Copyright
    
Copyright (c) 2018, Hyesu Jang
