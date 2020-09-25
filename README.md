
***respyte***
==============================
A python implementation of open-source version of RESP method

## Short description of respyte beta ver.
The main goals of the reformatting are (1) implementation of better user interface and (2) to support various charge models (fuzzy charge model, virtual site etc.)

Things that are currently not supported (but will be implemented soon!) are (1) normalization and (2) the usage of electric field term in fitting.

### 1. conda environment and dependent packages 
Create a new environment and install all dependencies
```
conda create -n respyte-beta python=3.7
source activate respyte-beta
conda install -c psi4 psi4 
conda install scipy networkx pyyaml 
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
respyte-esp_generator <input-dir-name>
```
#### 2.2. Charge fitting 

Once the ESP calculation is successfully done, you are ready to go to the next step. You can find the  output  files with `.espf` extension in each subfolder,`<input-dir-name>/molecules/<molecule name>/conf(j)`,  which  store all the grid point informations with a file format that the respyte package can understand.

Write `respyte.yml` inside `<input-dir-name>` and run this command:

```
respyte-optimizer <input-dir-name>
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
- `self.atomids` : list of atom ids (similar with atom type concept in general force field) 
- `self.atomid_dict` : dictionary storing information about each atom id
- For  example, atomids and atomid_dict for methanol molecule are: 
```
atomids = [3, 1, 2, 0, 0, 0]
atomid_dict = defaultdict(list,
{3: [{'molname': 'meoh', 'resname': 'MOL',  'atomname': 'O1', 'elem': 'O', 'vdw_radius': 2.8723237902580037}],
 1: [{'molname': 'meoh', 'resname': 'MOL', 'atomname': 'H2', 'elem': 'H', 'vdw_radius': 2.267624044940529}],
 2: [{'molname': 'meoh', 'resname': 'MOL',  'atomname': 'C3',  'elem': 'C', 'vdw_radius': 3.2124673969990827}],
 0: [{'molname': 'meoh', 'resname': 'MOL', 'atomname': 'H4', 'elem': 'H', 'vdw_radius': 2.267624044940529},
      {'molname': 'meoh', 'resname': 'MOL',  'atomname': 'H5', 'elem': 'H', 'vdw_radius': 2.267624044940529},
      {'molname': 'meoh', 'resname': 'MOL',  'atomname': 'H6',  'elem': 'H', 'vdw_radius': 2.267624044940529}]})
```
- `self.gridxyz`, `self.espval`, `self.efval` : information about grids and QM values
- Please note that all the units of length are converted into Bohr, not Angstrom.
- And **molecules object**  take multiple molecule objects and reassign atomids so that there’s no atom id used multiple times for different atoms. 
```
# create a respyte molecules object
molecules = respyte_molecules()
# add onle molecule object to molecules object
molecules.add_molecule(molecule) 
```
#### (2) `objective.py`
: **Objective object** mainly calculate an objective function, gradient and hessian of the objective function at a given point of parameter space. you can set the model as ‘point_charge’  or ‘fuzzy_charge’ And penalty is a dictionary containing information of additive penalty term in the objective function. One example is `penalty={'type': 'L1', 'a': 0.001, 'b': 0.1}`, which specifies hyperbolic charge restraint with a=0.001, b=0.1,  which is a weak restraining weight in 1993 paper. 

It stores the values (q and alpha) into `vals`(list) and their information into `val_info`(list). `val_info` stores atom id, model (‘point_charge’ or ‘fuzzy_charge’) and variable type(‘q’ or ‘alpha’). An example of `vals` and `val_info` is like this: 
```
vals = [0, 0, 0, 0, 1] 
val_info = [[0, 'point_charge', 'q'], [1, 'point_charge', 'q'], [2, 'point_charge', 'q'], [3, 'point_charge', 'q'], ['l0', 'point_charge', 'lambda']]
```
#### (3) `optimizer.py`
: **Optimizer object** take objective object and run Newton-Rapson method using scipy and return the values once the convergence criterion is  met. 

#### (4) `procedure.py`
: function called `resp` which run single-stage or two-stage fitting  procedure. 
An example of the usage of the function is:
```
print('example 1. analytic solution of single-stg-fit\n')
penalty =  {'type': 'L1', 'a': 0.001, 'b': 0.1}
resp(molecules, symmetry='all', model='point_charge', penalty=penalty, procedure=1)
```

#### Copyright
    
Copyright (c) 2018, Hyesu Jang
