import numpy as np
import copy
from collections import defaultdict, OrderedDict
import rdkit.Chem as rdchem

from respyte.fbmolecule import * 
from respyte.fbmolecule import Molecule as FBMolecule
from respyte.parse import *
from respyte.select_grid import SelectGridPts

# Global variable
bohr2Ang = 0.52918825 # converting Bohr to Angstrom

class respyte_molecule:
    """ Respyte molecule object 

    ...
    Attributes
    ----------
    name : str
        molecule name. 
    fbmol : FBMolecule
        ForceBalance molecule object. 
        This carries molecule information(geometry, elements, residue names, atom names).
    abspath  : str
        absolutepath of the input coordinate file.
    symmetryClass : list
        a list of symmetry classes of atoms 
    polar_atom_indices: list
        a list of polar atom indices.
    atomids: list
        a list of atom ids.
        (id = unique number assigned to each unique atom for informing 
            which atoms are equivalent during the charge fitting procedure)
    atomid_dict: dict
        a dictionary containing information of atom ids.
    input_equiv_atoms: list 
        a list of equivalent atoms user manually specify to force the same charge on the selected atoms
        [[atomname1, atomname2, ...], [resname1, resname2, ...]] 
    fixed_charges: list
        a list of [[atom indices], sum of charges of the atoms].
    gridxyz : list
        a list of grid xyz coordinates.
    espval : list
        a  list of electrostatic potential values.
    efval : list
        a list of electric field values.
    Methods
    -------
    read_coord_file(coord_fnm):
        read coordinate file and store attributes.
    reset_atom_id:
        reset atom ids using symmetry classes of atoms.
    update_atomid_dict:
        update atomid_dict based on the current atomids.
    set_atom_id(symmetry='polar'): 
        set atom ids and update atomids, atomid_dict.
    set_fragment_charge(list_of_atom_idx, charge): 
        set net charge of the fragment fixing during the fitting.
    set_residue_charge_by_resname(residue_name, charge):
        set net residue charge being fixed during the fitting.
    set_net_charge(charge):
        set net charge of the molecule.
    read_espf(espf_fnm, settings):
        read espf file and store information.
    GetAtoms:
        function for looping over atoms 
    """
    def __init__(self, molecule_name= None, coord_fnm=None, espf_fnm=None,  settings = None, input_equiv_atoms = []):
        '''
        Parameters:    
            molecule_name : str
                name of molecule.    
            coord_fnm : str
                mol2 or pdb file name containing molecule geometry
            espf_fnm : str
                espf file name
            settings : dict
                a dictionary containng grid selection settings 
                segttings =  {'mol': fbmolecule object, 'inner': inner boundary(float), 'outer': outer boundary(float), 'radiiType': radii type(str)}
        '''
        self.name  = molecule_name 
        self.input_equiv_atoms = input_equiv_atoms # should  be located before read  coord file

        if coord_fnm == None: 
            self.abspath = None
            self.fbmol = FBMolecule()
            self.polar_atom_indices = []
            self.atomids = []
            self.atomid_dict = {}
        else: 
            self.read_coord_file(coord_fnm)

        self.fixed_charges = []

        if espf_fnm == None:
            self.gridxyz = []
            self.espval  = []
            self.efval   = []
        else:
            self.read_espf(espf_fnm, settings)
    
    def add_input_equiv_atoms(self, input_equiv_atoms, reset=True): ##
        '''
        add a list of equivalent atoms user manually specify to force the same charge on the selected atoms

                Parameters:
                        input_equiv_atoms (list): list of equivalent atoms
        ''' 
        assert isinstance(input_equiv_atoms, list), 'input_equiv_atoms should be a list'
        for equiv_atoms in input_equiv_atoms: 
            assert len(equiv_atoms) == 2
        self.input_equiv_atoms = input_equiv_atoms

        self.reset_atom_id()
        
    def read_coord_file(self, coord_fnm):    
        '''
        read coordinate file and store attributes

                Parameters:
                        coord_fnm : mol2 or pdb file name
        ''' 
        # print(f'## coord_fnm: {coord_fnm}')
        self.abspath = os.path.abspath(coord_fnm)
        fbmol = FBMolecule(coord_fnm)
        fbmol.add_vdw_radii('bondi') # store vdw radii for fuzzy charge
        fbmol.xyzs = np.array(fbmol.xyzs) / bohr2Ang # switch unit to Bohr
        self.fbmol = fbmol

        # using rdkit to find polar atoms
        self.polar_atom_indices = get_polar_atom_indices(coord_fnm) 

        # storing symmetryclass instead of rdkit molecule object?
        self.symmetryClass = get_symmetry_class(coord_fnm)
        # print(f'## symmetryClass: {self.symmetryClass}')
        self.reset_atom_id()

    def reset_atom_id(self):
        '''
        reset atom ids using symmetry classes of atoms.
        ''' 

        sym_set = list(set(self.symmetryClass))
        sym_set.sort()
        # print(f'## sym_set: {sym_set}')
        atomids = np.zeros((len(self.symmetryClass)),dtype=int)
        for sym in sym_set:
            atomid = int(sym_set.index(sym))
            indices = [i for i, x in enumerate(self.symmetryClass) if x == sym]
            for i in indices:
                atomids[i] = atomid
        self.atomids = list(atomids)
        # print(f'## atomids: {self.atomids}')
        self.update_atomid_dict()

        # if self.input_equiv_atoms is not empty, force user-specified symmetry on selected atoms
        if len(self.input_equiv_atoms) !=0:
            new_atomids =  copy.deepcopy(self.atomids)
            for equiv_atoms in self.input_equiv_atoms: 
                atomnames, resnames  = equiv_atoms
                equiv_ids = []
                for atomid, lst_of_info in self.atomid_dict.items():
                    match = False
                    for info in lst_of_info:
                        if info['resname'] in resnames and info['atomname'] in atomnames:
                            match = True
                            break
                    if match:
                        equiv_ids.append(atomid)
                new_id = min(list(equiv_ids))
                new_atomids = [new_id if i in equiv_ids else i for i in new_atomids]

            self.atomids  =  new_atomids
            self.update_atomid_dict()
        
    def update_atomid_dict(self):
        '''
        update atomid_dict based on the current atomids.
        ''' 
        atomid_dict = defaultdict(list)
        for atomid, resname, atomname, elem, vdw_radius in zip(self.atomids, self.fbmol.resname, self.fbmol.atomname, self.fbmol.elem, self.fbmol.vdwradii):
            info = {'molname':self.name, 'resname': resname, 'atomname': atomname, 'elem':elem, 'vdw_radius': vdw_radius / bohr2Ang}
            if info not in atomid_dict[atomid]:
                atomid_dict[atomid].append(info)   
        self.atomid_dict =  atomid_dict

    def set_atom_id(self, symmetry='polar'):
        '''
        set atom ids and update atomids, atomid_dict.

                Parameters: 
                        symmetry (str): "all" , "polar" or "nosym"
                        note that when you set symmetry "polar" , or "nosym" part (or all) of 
                        the equivalent atoms you set in  the input file will be washed  out. 
        ''' 
        self.reset_atom_id() 
        if symmetry == 'nosym':
            new_atomids = list(range(len(self.atomids)))

        elif symmetry == 'polar':
            new_atomids = [i if idx in  self.polar_atom_indices else -1 for idx, i in enumerate(self.atomids)]
            nonpolar_new_id = np.amax(new_atomids)  + 1
            for idx, atomid in enumerate(new_atomids):
                if atomid < 0:
                    new_atomids[idx] = nonpolar_new_id
                    nonpolar_new_id += 1
        elif symmetry == 'all':
            new_atomids = self.atomids

        else: 
            raise NotImplementedError(f'symmetry={symmetry} is not implemented. "nosym", "polar" and "all" are available. ')
        self.atomids = new_atomids
        self.update_atomid_dict()

    def set_fragment_charge(self, list_of_atom_idx, charge):
        '''
        set net charge of the fragment being fixed during the fitting.

                Parameters:
                        list_of_atom_idx (list): a list of atom indices included in the fragment
                        charge (float): net  charge of the fragment
        ''' 
        assert isinstance(list_of_atom_idx, list), f'Wrong type of input list.'
        assert all(isinstance(atom_idx, int) for atom_idx in list_of_atom_idx), f'Element type of input list should be integer'
        assert len(list_of_atom_idx) == len(set(list_of_atom_idx)), f'Duplicates in input list.'
        assert isinstance(charge, (int, float)), f'Net charge should be a number. wrong input charge: {charge}'
        self.fixed_charges.append([list_of_atom_idx, charge])

    def set_residue_charge_by_resname(self, residue_name, charge):
        '''
        set net residue charge being fixed during the fitting.

                Parameters:
                        residue_name (str): residue name
                        charge (float): net  charge of the residue
        ''' 
        assert isinstance(charge, (int, float)), f'Net charge should be a number. wrong input charge: {charge}'
        list_of_atom_idx = []
        for idx, atom in enumerate(self.GetAtoms()):
            if atom.resname == residue_name:
                list_of_atom_idx.append(idx)
        self.fixed_charges.append([list_of_atom_idx, charge])

    def set_net_charge(self, charge):
        '''
        set net charge of the molecule.

                Parameters:
                        charge (float): net  charge of the molecule
        ''' 
        assert isinstance(charge, (int, float)), f'Net charge should be a number. wrong input charge: {charge}'
        list_of_atom_idx = list(range(len(self.GetAtoms())))
        self.fixed_charges.append([list_of_atom_idx, charge])


    def read_espf(self,espf_fnm, settings):
        '''
        read espf file and store information.

                Parameters:
                        espf_fnm (str): espf file name
                        settings (dict): a dictionary containng grid selection settings 
        ''' 
        if settings  == None:  
            selectedPtsIdx = None
        else:  
            pts = []
            with open(espf_fnm, 'r') as espff:
                for i, line in enumerate(espff):
                    fields = line.strip().split()
                    numbers = [float(field) for field in fields]
                    if (len(numbers)==4):
                        xyz = [x for x in numbers[0:3]]
                        pts.append(xyz)
            selectedPtsIdx, selectedPts = SelectGridPts(pts, settings) 
            print(f"  * selected {len(selectedPtsIdx)} pts out of {len(pts)} pts") 

        gridxyz = []
        espval = []
        efval = []

        selectedLines = []
        with open(espf_fnm, 'r') as espff:
            if selectedPtsIdx == None: 
                for i in range(len(espff.readlines())):
                    selectedLines.append(i)
            else: 
                for i in selectedPtsIdx:
                    selectedLines.append(int(i*2))
                    selectedLines.append(int(i*2+1))                    
                
        with open(espf_fnm, 'r') as espff:
            for i, line in enumerate(espff):
                if  i in selectedLines:
                    fields = line.strip().split()
                    numbers = [float(field) for field in fields]
                    if (len(numbers)==4):
                        xyz = [x/bohr2Ang for x in numbers[0:3]]
                        gridxyz.append(xyz)
                        espval.append(numbers[3])
                    elif (len(numbers)==3):
                        efval.append(numbers[0:3])
        self.gridxyz = gridxyz
        self.espval  = espval
        self.efvals  = efval  

    def GetAtoms(self):
        atoms = []
        for idx, atomid in enumerate(self.atomids):
            atom = respyte_atom(self.fbmol, idx, atomid)
            atoms.append(atom)
        return atoms

class respyte_atom:
    """ An atom class in respyte package. 

    ...
    Attributes
    ----------
    idx : int
        index of atom.
    xyz : list
        list of xyz  coordinates in Bohr.
    atomname : str
        atom name given by input coordinate file.
    resname: str
        residue name given by input coordinate file.
    resid: int
        residue number
    elem: str
        atomic symbol
    id: int
        atom id
    """
    def __init__(self, fbmol, idx, atomid):
        self.idx = idx
        self.xyz = fbmol.xyzs[0][idx]
        self.atomname = fbmol.atomname[idx]
        self.resname  = fbmol.resname[idx]
        self.resid    = fbmol.resid[idx]
        self.elem     = fbmol.elem[idx]
        self.id = atomid

class respyte_molecules:
    """ Respyte molecule object 

    ...
    Attributes
    ----------
    mols : list
        a list of respyte_molecule objects
    atomids: list
        a list of atom ids.
    atomid_dict: dict
        a dictionary containing information of atom ids.
    fixed_charges: list
        a list of [[atom ids], sum of charges of the atoms].
    Methods
    -------
    add_molecule(molecule):
        add respyte_molecule object.
    change_idx_to_id_fixed_charges(fixed_charges, atomids):
        change list of indices in fixed_charge to list of ids. 
    find_id(info, atomid_dict):
        find id from given dictionary.
    update_atomids(current_atomids, current_atomid_dict, input_molecule)
         update atomids and atomid_dict of the input molecule object. 
    set_atom_id(symmetry='polar'):
        set atom ids and update atomids, atomid_dict.
    get_polar_ids:
        return a list of polar atom ids.
    fix_polar_charges_from_previous_step(object)
    """

    def __init__(self):
        self.mols = []
        self.atomids = []
        self.atomid_dict = {}
        self.fixed_charges = []

    def add_molecule(self, molecule):
        '''
        add respyte_molecule object. 

                Parameters:
                        molecule (respyte_molecule): respyte_molecule object       
        ''' 
        assert isinstance(molecule, respyte_molecule), 'input molecule is not a respyte_molecule'

        # 1.update ids to avoid overlap between different sets of atom ids from each molecule
        self.update_atomids(self.atomids, self.atomid_dict, molecule)

        # 2. save molecule into mols 
        self.mols.append(molecule)

        # 3. update self.atomids, self.atomid_dict 
        self.atomid_dict.update(molecule.atomid_dict)
        self.atomids = list(set(self.atomid_dict.keys()))

        # 4. update fixed_charges -> [[ids], charge], ...]
        new_fixed_charges = self.change_idx_to_id_fixed_charges(molecule.fixed_charges, molecule.atomids)
        for new_fixed_charge_lst in new_fixed_charges:
            if new_fixed_charge_lst not in self.fixed_charges:
                self.fixed_charges.append(new_fixed_charge_lst)

    def from_input(self, inp):
        '''
        add molecule objects by reading respyte input object. 

                Parameters:
                        inp (Input): Input object       
        ''' 
        input_equiv_atoms = inp.equiv_atoms

        for name, info in inp.mols.items():
            nconf = info['nconf']
            net_charge = info['net_charge']
            for j in range(1, nconf+1):
                confN = 'conf%d' % (j)
                path = os.path.join(inp.inp_dir, 'molecules', name, confN)
                molN_confN = '%s_%s' % (name, confN)
                pdbfile = os.path.join(path, '%s.pdb' % molN_confN)
                mol2file = os.path.join(path, '%s.mol2' % molN_confN)
                if os.path.isfile(pdbfile):
                    coord_file = pdbfile
                elif os.path.isfile(mol2file):
                    coord_file = mol2file
                else:
                    raise RuntimeError(" Coordinate file should have pdb or mol2 file format! ")
                espf_file = os.path.join(path, '%s.espf' % molN_confN)
                # molecule may want to store its name. 
                # need to implement function store selected grid pts (later)
                if inp.gridinfo == None:
                    settings = None
                else:
                    settings = {'mol': FBMolecule(coord_file), 'inner': inp.gridinfo['inner'], 'outer':inp.gridinfo['outer'], 'radiiType': inp.gridinfo['radii']}
                    print(f"  * grid selection setting: {{inner: {inp.gridinfo['inner']}, outer:{inp.gridinfo['outer']}, radiiType: {inp.gridinfo['radii']}}}")
                mol = respyte_molecule(name, coord_file, espf_file, settings,  input_equiv_atoms)
                
                # net charge
                mol.set_net_charge(net_charge)
                # residue charge/ fixed charge 
                resnames = list(set(mol.fbmol.resname))
                for resname in resnames:
                    if resname in list(inp.resChargeDict.keys()):
                        mol.set_residue_charge_by_resname(resname, inp.resChargeDict[resname])
                for atomN, fixed_atomic_charge_info in inp.fixed_atomic_charge.items():
                    for atom in mol.GetAtoms():
                        if atom.resname == fixed_atomic_charge_info['resname'] and atom.atomname == fixed_atomic_charge_info['atomname']:
                            mol.set_fragment_charge([atom.idx], fixed_atomic_charge_info["charge"])

                self.add_molecule(mol)

    def change_idx_to_id_fixed_charges(self, fixed_charges, atomids):
        '''
        change list of indices in fixed_charge to list of ids.  

                Parameters:
                        fixed_charges (list): a list  of [list of indices ,fixed_charge]
                        atomids (list): a list of atom ids     
        ''' 
        new_fixed_charges = []
        for list_of_atom_idx, charge in fixed_charges:
            list_of_atom_ids = [atomid for idx, atomid in enumerate(atomids) if idx in list_of_atom_idx]
            new_fixed_charges.append([list_of_atom_ids, charge])
        return new_fixed_charges

    def find_id(self, info, atomid_dict):
        '''
        search id from a dictionary.

                Parameters:
                        info (dict): a dictionary containing information of atom id of interest
                        atomid_dict (dict): a dictionary containing information of atom ids.

                Returns:
                        atomid_matched  (int): atom id matching to the info
        ''' 
        atomid_matched = None
        for atomid, list_of_info in atomid_dict.items():
            if info in list_of_info:
                atomid_matched = atomid
        return atomid_matched
    
    def update_atomids(self, current_atomids, current_atomid_dict, input_molecule):
        '''
        update atomids and atomid_dict of the input molecule object. 

                Parameters:
                        current_atomids (list): a list of atom ids.
                        current_atomid_dict (dict): a dictionary containing information of atom ids.
                        input_molecule (respyte_molecule): respyte_molecule object 
        ''' 
        if len(current_atomids) == 0:
            pass
        else: 
            starting_number = max(current_atomids) +1 
            new_atomids = list(np.array(input_molecule.atomids)+ starting_number)
            input_molecule.atomids = new_atomids
            input_molecule.update_atomid_dict()

            for atomid, list_of_info in input_molecule.atomid_dict.items():
                for info in list_of_info:
                    atomid_matched = self.find_id(info, current_atomid_dict)
                    if atomid_matched is not None: 
                        break 
                if atomid_matched is not None: 
                    input_molecule.atomids = [atomid_matched if x == atomid else x for x in input_molecule.atomids]
            input_molecule.update_atomid_dict()
        
    def set_atom_id(self, symmetry='polar'):
        '''
        set atom ids and update atomids, atomid_dict.

                Parameters: 
                        symmetry (str): "all" , "polar" or "nosym"
        '''    
        new_atomids = []
        new_atomid_dict ={}
        new_fixed_charges = []
        # loop over molecules and reassign ids 
        for mol in self.mols: 
            # update mol first 
            mol.set_atom_id(symmetry)
            self.update_atomids(new_atomids, new_atomid_dict, mol) 
            # then update atomids, atomid_dict, fixed_charges? 
            new_atomid_dict.update(mol.atomid_dict)
            new_atomids = list(set(new_atomid_dict.keys()))
            fixed_charges_in_ids = self.change_idx_to_id_fixed_charges(mol.fixed_charges, mol.atomids)
            for fixed_charge_lst in fixed_charges_in_ids:
                if fixed_charge_lst not in new_fixed_charges: 
                    new_fixed_charges.append(fixed_charge_lst)
        self.atomids = new_atomids
        self.atomid_dict = new_atomid_dict
        self.fixed_charges = new_fixed_charges

    def get_polar_ids(self):
        '''
        return a list of polar atom ids.

                Returns:
                        polar_ids (list): a list of polar atom ids.
        '''    
        polar_ids =  []
        for mol in self.mols: 
            for index in mol.polar_atom_indices:
                polar_ids.append(mol.atomids[index])
        polar_ids = list(set(polar_ids))
        return polar_ids

    def fix_polar_charges_from_previous_step(self, objective):
        '''
        add polar atom information to fixed_charges.

                Parameters: 
                        objective (respyte_objective): objective object containing charges
        '''    
        # processing objective object
        charges = {} 
        for val, val_info in zip(objective.vals, objective.val_info):
            atomid_, model, vartype = val_info
            if vartype == 'q':
                #current_id<-atominfo <-atomid
                atomid_info_ = objective.molecules.atomid_dict[atomid_]
                
                atomid = self.find_id(atomid_info_[0], self.atomid_dict)
                charges[atomid] = val

        polar_ids = self.get_polar_ids()
        for polar_id in  polar_ids:
            polar_charge = charges[polar_id]
            self.fixed_charges.append([[polar_id], polar_charge])


def rdmolFromFile(filename):
    '''
    Read mol2 or pdb file and return a rdkit molecule object

            Parameters:
                    filename (str): mol2 or pdb file name

            Returns:
                    mol : rdkit molecule object
    '''
    if filename.endswith('.mol2'):
        mol = rdchem.MolFromMol2File(filename, removeHs=False)
    elif filename.endswith('.pdb'):
        mol = rdchem.MolFromPDBFile(filename, removeHs=False)
    else:
        raise RuntimeError('The extension of input file should be either mol2 or pdb!')
    return mol

def assignRadiiRdmol(mol, radii='bondi'):
    '''
    Assign vdW radii to input rdkit molecule object using SetProp function.

            Parameters:
                    mol : rdkit molecule object
                    radii (str): radii type. 'bondi' or 'Alvarez'

            Returns:
                    mol : rdkit molecule object with radii assigned 
    '''
    if radii =='bondi':
        for atom in mol.GetAtoms():
            atom.SetProp('radius',str(BondiRadii[atom.GetAtomicNum()-1])) # atom.GetProp('radius')
    elif radii =='Alvarez':
        for atom in mol.GetAtoms():
            atom.SetProp('radius',str(AlvarezRadii[atom.GetAtomicNum()-1]))
    else:
        raise NotImplementedError('assigning modified bondi radii on RDKit mol not implemented yet!')
    return mol
    
def get_polar_atom_indices(filename):
    '''
    Read mol2 or pdb file and return a list of polar atom indices

            Parameters:
                    filename (str): mol2 or pdb file name

            Returns:
                    polar_atom_indices (list): a list of polar atom indices
    '''
    rdmol = rdmolFromFile(filename)
    polar_atom_indices = []
    # 1. non-alkyl carbons
    for atom in rdmol.GetAtoms():
        if atom.GetSymbol() == 'C' and str(atom.GetHybridization()) != 'SP3':
            polar_atom_indices.append(atom.GetIdx())
    for atom in rdmol.GetAtoms():
        # 2. non-alkyl hydrogens
        if atom.GetSymbol() == 'H':
            for bond in atom.GetBonds():
                atom2 = bond.GetOtherAtom(atom)
                if (atom2.GetSymbol() != 'C' and atom2.GetSymbol() !='H'):
                    polar_atom_indices.append(atom.GetIdx())
                elif atom2.GetSymbol() == 'C' and atom2.GetIdx() in polar_atom_indices:
                    polar_atom_indices.append(atom.GetIdx())
        # 3. atoms besides C and H
        if (atom.GetSymbol() != 'C' and atom.GetSymbol() !='H'):
            polar_atom_indices.append(atom.GetIdx())
    return polar_atom_indices

def get_symmetry_class_using_CIPRank(filename):
    '''
    Read mol2 or pdb file and return a list of CIP ranks of atoms

            Parameters:
                    filename (str): mol2 or pdb file name

            Returns:
                    symmetryClass (list): a list of CIP ranks of atoms
    '''
    rdmol = rdmolFromFile(filename)
    rdchem.AssignStereochemistry(rdmol, cleanIt=True, force=True, flagPossibleStereoCenters=True)
    symmetryClass = []
    for atom in rdmol.GetAtoms():
        symmetryClass.append(int(atom.GetProp('_CIPRank')))
    return symmetryClass

def get_symmetry_class(filename):
    '''
    Read mol2 or pdb file and return a list of symmetry classes using CanonicalRankAtoms

            Parameters:
                    filename (str): mol2 or pdb file name

            Returns:
                    symmetryClass (list): a list of symmetry classes
    '''
    rdmol = rdmolFromFile(filename)
    symmetryClass = list(rdchem.CanonicalRankAtoms(rdmol, breakTies=False))
    return symmetryClass

                
