
***respyte***
==============================
A python implementation of open-source version of RESP method

## Short description of respyte beta ver.
The main goals of the reformatting are (1) implementation of better user interface and (2) to support various charge models (fuzzy charge model, virtual site etc.)

Things that are currently not supported (but will be implemented soon!) are (1) normalization and (2) the usage of electric field term in fitting.

### 1. conda environment and dependent packages 

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
And run `setup.py` for installation of the main package:
```
python setup.py install
```

### 2. Command-line interface 
Before running `esp_generator.py`, Please navigate `/data/input.sample` which contains sample of input folder and files. This will give you some idea on the data structure the package can handle.

#### 2.1. ESP calculation using Psi4
First, to calculate electrostatic potential (and electric field) using Psi4, create input folder with an appropriate data structure and run this command:

```
respyte-esp_generator <input-dir-name>
```
#### 2.2. Charge  fitting 

Once the ESP calculation is successfully done, you are ready to go to the next step. You can find the  output  files with `.espf` extension in each subfolder,`<input-dir-name>/molecules/<molecule name>/conf(j)`,  which  store all the grid point informations with a file format that the respyte package can understand.

Write `respyte.yml` inside `<input-dir-name>` and run this command:

```
respyte-optimizer <input-dir-name>
```

### 3. Using respyte as a python module

example of ipynb can  be found in  `/data/ipynb_example/example.ipynb`. 
    
To  use  the package as a python module, there are three main components you may want to know:

#### `(1) molecules.py`
: Creates **molecule object**, which takes coordinate file and espf file and store data for charge fitting. The way to generate molecule object is: 
```
# generate molecule object
molecule = respyte_molecule(molecule_name= 'meoh', coord_fnm='test.pdb', espf_fnm='test.espf')
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
- Please note that all the units for coordinates are converted into Bohr, not Angstrom.
- And molecules object  take multiple molecule object and reassign atomids so that there’s no atom id used multiple times for different atoms. 
```
molecules = respyte_molecules()
molecules.add_molecule(molecule)
```
#### (2) `objective.py`
: Objective object mainly calculate objective function, gradient and hessian at a given point of parameter space. you can set the model as ‘point_charge’  or ‘fuzzy_charge’ And penalty is a dictionary containing  information of penalty function. One example is `penalty={'type': 'L2', 'a': 0.001, 'b': 0.1}` and this specifies hyperbolic charge restraint with a=0.001, b=0.1,  which is a  weak restraining weight. 

I tried to  store the values (q and alpha) into `vals`(list) and their information into `val_info`(list). val_info stores atom id, model (‘point_charge’ or ‘fuzzy_charge’) and variable type(‘q’ or ‘alpha’). Example of vals and val_info are like this: 
```
vals = [0, 0, 0, 0, 1] 
val_info = [[0, 'point_charge', 'q'], [1, 'point_charge', 'q'], [2, 'point_charge', 'q'], [3, 'point_charge', 'q'], ['l0', 'point_charge', 'lambda']]
```
#### (3) `optimizer.py`
: Optimizer object take objective object and run Newton-Rapson method using scipy and return the values with the information once it is converged. 

#### (4) `procedure.py`
: function called `resp` which run single-stage or two-stage fitting  procedure. 
Example of the usage of the  function is:
```
print('example 1. analytic solution of single-stg-fit\n')
penalty =  {'type': 'L2', 'a': 0.001, 'b': 0.1}
resp(molecules, symmetry='all', model='point_charge', penalty=penalty, procedure=1)
```

#### Copyright
    
Copyright (c) 2020, Hyesu Jang
