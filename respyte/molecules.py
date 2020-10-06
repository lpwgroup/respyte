import os
import numpy as np
import copy
from collections import defaultdict, OrderedDict
from warnings import warn
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
    polar_hydrogen_indices: list
        a list of polar hydrogen atom indices.
    atom_equiv : dict
        a dictionary, whose keys are equivalence level('nosym', 'connectivity', 'relaxed_connectivity', 'symbol', 'symbol2')
        and the values are dictionarys, having 'equivs', a list of equiv values for the corresponding equivalence level, 
        and 'info', a dictionary contains definition of each equiv value. 
        note that term, 'equiv' is the replacement of 'atomid'
    input_equiv_atoms: list 
        a list of equivalent atoms user manually specify to force the same charge on the selected atoms
        [[molname1, molname2, ...],[atomname1, atomname2, ...], [resname1, resname2, ...]] 
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
    add_input_equiv_atoms(input_equiv_atoms, update=True):
        add user-specified equivalent atom list and update atom_equiv if update==True.
    read_coord_file(coord_fnm):
        read coordinate file and add attributes.
    get_atom_equiv(reset=False):
        create atom_equiv dictionary, if reset==True, remove information of user-specified symmetry and restore atom_equiv.
    reset_atom_equiv:
        remove user-specified symmetry and restore atom_equiv.
    get_atom_equiv_info(equivs):
        using the input list of equiv values, generate a dictionary of equiv value definitions.
    update_atom_equiv_info(equivs):
        apply input_equiv_atoms(user-specified equivalent atom list) and regenerate equivs and info.
    convert_index_to_equiv(indices, equiv_level):
        return a list(or single int) of equiv values assigned to input indices in a given equivalence level.
    set_fragment_charge(list_of_atom_idx, charge): 
        set net charge of the fragment fixing during the fitting.
    set_residue_charge_by_resname(residue_name, charge):
        set net residue charge being fixed during the fitting.
    set_net_charge(charge):
        set net charge of the molecule.
    read_espf(espf_fnm, settings):
        read espf file and store information.
    GetAtoms:
        return list of atoms
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
            input_equiv_atoms: list 
                a list of equivalent atoms user manually specify to force the same charge on the selected atoms
                [[[molname1, molname2, ...],[atomname1, atomname2, ...], [resname1, resname2, ...]] , ...]
        '''
        if molecule_name is None: 
            warn('molecule name is not provided. Use default molecule name, MOL. It may mess up atom equivalence between different molecules!')
            molecule_name = 'MOL'
        self.name = molecule_name
        
        self.add_input_equiv_atoms(input_equiv_atoms, update=False)

        if coord_fnm == None: 
            self.abspath = None
            self.fbmol = FBMolecule()
            self.polar_atom_indices = []
            self.polar_hydrogen_indices = []
            self.symmetryClass = []
            self.atom_equiv = {}
        else: 
            self.read_coord_file(coord_fnm)

        self.fixed_charges = []

        if espf_fnm == None:
            self.gridxyz = []
            self.espval  = []
            self.efval   = []
        else:
            self.read_espf(espf_fnm, settings)

    def add_input_equiv_atoms(self, input_equiv_atoms, update=True):
        '''
        add a list of equivalent atoms user manually specify to force the same charge on the selected atoms

                Parameters:
                        input_equiv_atoms (list): a list of equivalent atoms user manually specify to force the same charge on the selected atoms
        ''' 
        assert isinstance(input_equiv_atoms, list), 'input_equiv_atoms should be a list'
        for equiv_atoms in input_equiv_atoms: 
            assert len(equiv_atoms) == 3, 'each element of input_equiv_atoms should be like [[molname1, molname2, ...],[atomname1, atomname2, ...], [resname1, resname2, ...]]'    
        self.input_equiv_atoms = input_equiv_atoms
        if update: 
            self.get_atom_equiv()
    
    def read_coord_file(self, coord_fnm):
        '''
        read coordinate file and store attributes

                Parameters:
                        coord_fnm : mol2 or pdb file name
        ''' 
        self.abspath = os.path.abspath(coord_fnm)
        # using forcebalance molecule object to store basic information of the input conformer.
        fbmol = FBMolecule(coord_fnm)
        fbmol.add_vdw_radii('bondi') # store vdw radii (Angstrom) for fuzzy charge
        self.fbmol = fbmol

        # using rdkit tools, get atomic properties (## not sure if i want to keep them or not)
        self.polar_atom_indices, self.polar_hydrogen_indices, self.symmetryClass = get_atomic_properties(coord_fnm)

        # create atom_equiv dictionary from the informations 
        self.get_atom_equiv()

    def get_atom_equiv(self, reset=False):
        '''
        generate atom_equiv dictionary, which contains symmetry information

                Parameters:
                        reset (bool): if True, will remove user-forced symmetry and reset self.atom_equiv using canonical symmetry
        ''' 
        atom_equiv = {}
        # 1. no symmetry, 'nosym'
        nosym_equivs = list(range(len(self.fbmol.elem)))
        if reset: 
            info = self.get_atom_equiv_info(nosym_equivs)
            atom_equiv['nosym'] = {'equivs': nosym_equivs, 'info': info}
        else: 
            updated_nosym_equivs, updated_info = self.update_atom_equiv_info(nosym_equivs)
            atom_equiv['nosym'] = {'equivs': updated_nosym_equivs, 'info': updated_info}
        
        # 2. canonical symmetry, 'connectivity'
        sym_set= list(set(self.symmetryClass))
        sym_set.sort()
        connectivity_equivs = np.zeros((len(self.symmetryClass)), dtype=int)
        for sym in sym_set:
            equiv = int(sym_set.index(sym))
            indices = [i for i, x in enumerate(self.symmetryClass) if x== sym]
            for idx in indices: 
                connectivity_equivs[idx] = equiv
        connectivity_equivs = list(connectivity_equivs)
        if reset : 
            info = self.get_atom_equiv_info(connectivity_equivs)
            atom_equiv['connectivity'] = {'equivs': connectivity_equivs, 'info': info}
        else: 
            updated_connectivity_equivs, updated_info = self.update_atom_equiv_info(connectivity_equivs)
            atom_equiv['connectivity'] = {'equivs': updated_connectivity_equivs, 'info': updated_info}

        # 3. relaxed symmetry (force symmetry on polar atoms only), 'relaxed_connectivity'
        relaxed_equivs = [i if idx in self.polar_atom_indices else -1 for idx, i in enumerate(connectivity_equivs)]
        nonpolar_new_equiv = np.amax(relaxed_equivs) + 1 
        for idx, equiv in enumerate(relaxed_equivs):
            if equiv < 0: 
                relaxed_equivs[idx] = nonpolar_new_equiv
                nonpolar_new_equiv += 1
        if reset : 
            info = self.get_atom_equiv_info(relaxed_equivs)
            atom_equiv['relaxed_connectivity'] = {'equivs': relaxed_equivs, 'info': info}
        else: 
            updated_relaxed_equivs, updated_info = self.update_atom_equiv_info(relaxed_equivs)
            atom_equiv['relaxed_connectivity'] = {'equivs': updated_relaxed_equivs, 'info': updated_info}
        
        # 4. atomic numbers, 'symbol'
        symbol_equivs = []
        for elem in self.fbmol.elem:
            atomic_number = list(PeriodicTable.keys()).index(elem) + 1 
            symbol_equivs.append(atomic_number)
        info = self.get_atom_equiv_info(symbol_equivs)
        atom_equiv['symbol'] = {'equivs': symbol_equivs, 'info': info}
         
        # 5. atomic numbers with -1 for polar hydrogen, 'symbol2'
        symbol2_equivs = copy.deepcopy(symbol_equivs)
        for idx in self.polar_hydrogen_indices: 
            symbol2_equivs[idx] = -1
        info = self.get_atom_equiv_info(symbol2_equivs)
        atom_equiv['symbol2'] = {'equivs': symbol2_equivs, 'info': info}

        self.atom_equiv = atom_equiv

    def reset_atom_equiv(self):
        '''
        remove user-specified symmetry and reset self.atom_equiv using canonical symmetry
        '''
        self.get_atom_equiv(reset=True)
    
    def get_atom_equiv_info(self, equivs):
        '''
        using the input list of equiv values, generate a dictionary of equiv value definitions
        '''
        atom_equiv_info = defaultdict(list)
        for equiv, resname, atomname, elem, vdw_radius in zip(equivs, self.fbmol.resname, self.fbmol.atomname, self.fbmol.elem, self.fbmol.vdwradii):
            info = {'molname':self.name, 'resname': resname, 'atomname': atomname, 'elem':elem, 'vdw_radius': vdw_radius}
            if info not in atom_equiv_info[equiv]:
                atom_equiv_info[equiv].append(info)
        return atom_equiv_info

    def update_atom_equiv_info(self, equivs):
        '''
        apply self.input_equiv_atoms(user-specified equivalent atom list) and return new equivs and info
        '''
        atom_equiv_info = self.get_atom_equiv_info(equivs)
        new_equivs = copy.deepcopy(equivs)
        for equiv_atoms in self.input_equiv_atoms: 
            molnames, atomnames, resnames = equiv_atoms 
            same_equivs = []
            for equiv, lst_of_info in atom_equiv_info.items():
                match = False
                for info in lst_of_info:
                    if info['molname'] in molnames and info['resname'] in resnames and info['atomname'] in atomnames:
                        match = True
                        break
                if match: 
                    same_equivs.append(equiv)
            if len(same_equivs) > 0:
                new_equiv = min(list(same_equivs))
                new_equivs = [new_equiv if i in same_equivs else i for i in new_equivs]
        new_atom_equiv_info = self.get_atom_equiv_info(new_equivs)      
        return new_equivs, new_atom_equiv_info

    def convert_index_to_equiv(self, indices, equiv_level):
        '''
        return a list(or single int) of equiv values assigned to input indices in a given equivalence level
        '''
        if isinstance(indices, list):
            equivs_converted = [equiv for idx, equiv in enumerate(self.atom_equiv[equiv_level]['equivs']) if idx in indices]
        elif isinstance(indices, int): 
            equivs_converted = self.atom_equiv[equiv_level]['equivs'][indices]
        return equivs_converted

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
        list_of_atom_idx.sort()
        if [list_of_atom_idx, charge] not in self.fixed_charges:
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
        for  atom in self.GetAtoms():
            if atom.resname == residue_name:
                list_of_atom_idx.append(atom.idx)
        list_of_atom_idx.sort()
        if [list_of_atom_idx, charge] not in self.fixed_charges:
            self.fixed_charges.append([list_of_atom_idx, charge])
        
    def set_net_charge(self, charge):
        '''
        set net charge of the molecule.

                Parameters:
                        charge (float): net  charge of the molecule
        ''' 
        assert isinstance(charge, (int, float)), f'Net charge should be a number. wrong input charge: {charge}'
        list_of_atom_idx = list(range(len(self.GetAtoms())))
        list_of_atom_idx.sort()
        if [list_of_atom_idx, charge] not in self.fixed_charges:
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
                        xyz = [x for x in numbers[0:3]] # unit = Angstrom
                        gridxyz.append(xyz)
                        espval.append(numbers[3])
                    elif (len(numbers)==3):
                        efval.append(numbers[0:3])
        self.gridxyz = gridxyz
        self.espval  = espval
        self.efvals  = efval  

    def GetAtoms(self):
        atoms = []
        for idx, elem in enumerate(self.fbmol.elem):
            atom = respyte_atom(self.fbmol, self.atom_equiv, idx)
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
        list of xyz  coordinates in Angstrom.
    atomname : str
        atom name given by input coordinate file.
    resname: str
        residue name given by input coordinate file.
    resid: int
        residue number
    elem: str
        atomic symbol
    atom_equiv: dict
        a dictionary, whose keys are equivalence level('nosym', 'connectivity', 'relaxed_connectivity', 'symbol', 'symbol2')
        and the values are the equiv value assigned to the atom        
    """
    def __init__(self, fbmol, atom_equiv, idx):
        self.idx = idx
        self.xyz = fbmol.xyzs[0][idx]
        self.atomname = fbmol.atomname[idx]
        self.resname  = fbmol.resname[idx]
        self.resid    = fbmol.resid[idx]
        self.elem     = fbmol.elem[idx]
        self.atom_equiv = self.get_equiv(atom_equiv, idx)    

    def get_equiv(self, atom_equiv, idx):
        Answer = {}
        for equiv_level, dic in atom_equiv.items():
            equivs = dic['equivs']
            equiv = equivs[idx]
            Answer[equiv_level] = equiv
        return Answer

class respyte_molecules:
    """ Respyte molecule object 

    ...
    Attributes
    ----------
    mols : list
        a list of respyte_molecule objects
    atom_equiv : dict
        a dictionary, whose keys are equivalence level('nosym', 'connectivity', 'relaxed_connectivity', 'symbol', 'symbol2')
        and the values are dictionarys, having 'equivs', a list of set of equiv values used in the system(molecules object), 
        and 'info', a dictionary contains definition of each equiv value. 
        note that term, 'equiv' is the replacement of 'atomid'
    Methods
    -------
    add_molecule(molecule):
        add respyte_molecule object.
    update_molecule_atom_equiv(molecule):
        update equiv values to avoid the case where one equiv value is assigned to 
        non-equivalent atoms across different molecules.
    find_id(info, atom_equiv_info):
        search id from a dictionary.
    update_atom_equiv:
        update self.atom_equiv
    from_input(inp):
        add molecules from respyte input object
    """
    def __init__(self):
        self.mols = []
        self.atom_equiv = {}

    def add_molecule(self, molecule):
        '''
        add respyte_molecule object. 

                Parameters:
                        molecule (respyte_molecule): respyte_molecule object       
        ''' 
        assert isinstance(molecule, respyte_molecule), 'input molecule is not a respyte_molecule'
        # 1.update atom_equiv of the input molecule before adding it
        molecule = self.update_molecule_atom_equiv(molecule)
        # 2. store the molecule into mols 
        self.mols.append(molecule)
        # 3. update self.atom_equiv
        self.update_atom_equiv()

    def update_molecule_atom_equiv(self, molecule):
        '''
        update equiv values to avoid the case where one equiv value is assigned to 
        non-equivalent atoms across different molecules
        '''
        updated_molecule = copy.deepcopy(molecule)
        if len(self.mols) >0:
            for equiv_level, dic in updated_molecule.atom_equiv.items():
                if equiv_level in ['symbol', 'symbol2']:
                    pass 
                else: 
                    starting_number = max(self.atom_equiv[equiv_level]['equivs']) + 1
                    new_equivs = list(np.array(dic['equivs'])+ starting_number)
                    new_atom_equiv_info = updated_molecule.get_atom_equiv_info(new_equivs)

                    for equiv, list_of_info in new_atom_equiv_info.items():
                        for info in list_of_info:
                            equiv_matched = self.find_id(info, self.atom_equiv[equiv_level]['info'])
                            if equiv_matched is not None: 
                                break
                        if equiv_matched is not None: 
                            new_equivs = [equiv_matched if x == equiv else x for x in new_equivs]
                    new_atom_equiv_info = molecule.get_atom_equiv_info(new_equivs)
                    updated_molecule.atom_equiv[equiv_level] = {'equivs': new_equivs, 'info': new_atom_equiv_info}
        return updated_molecule

    def find_id(self, info, atom_equiv_info):
        '''
        search id from a dictionary.

                Parameters:
                        info (dict): a dictionary containing information of atom id of interest
                        atom_equiv_info (dict): a dictionary containing information of equivs.

                Returns:
                        atomid_matched  (int): equiv value matching to the info
        ''' 
        equiv_matched = None
        for equiv, list_of_info in atom_equiv_info.items():
            if info in list_of_info:
                equiv_matched = equiv
        return equiv_matched

    def update_atom_equiv(self):
        '''
        update self.atom_equiv
        '''
        atom_equiv = {}
        for mol in self.mols:
            for equiv_level, dic in mol.atom_equiv.items():
                equivs = dic['equivs']
                info   = dic['info']
                if equiv_level not in atom_equiv.keys():
                    atom_equiv[equiv_level] = defaultdict(dict)
                atom_equiv[equiv_level]['info'].update(info)
        for equiv_level in atom_equiv.keys():
            atom_equiv[equiv_level]['equivs'] = sorted(list(set(atom_equiv[equiv_level]['info'].keys())))
        self.atom_equiv = atom_equiv

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
                        if atom.resname == fixed_atomic_charge_info['resname'] and atom.atomname == fixed_atomic_charge_info['atomname']: ## do i want to check molname?
                            mol.set_fragment_charge([atom.idx], fixed_atomic_charge_info["charge"])

                self.add_molecule(mol)

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
    
def get_atomic_properties(filename):
    '''
    Using RDKit, get atomic polarities and symmetry classes of a molecule
            Parameters: 
                    filename (str): mo2 or pdb file name
            Returns: 
                    polar_atom_indices (list): a list of polar atom indices
                    polar_hydrogen_indices (list): a list of polar hydrogen indices
                    symmetryClass (list): a list of symmetry classes 

    '''
    rdmol = rdmolFromFile(filename)

    # 1. polar atom indices and polar hydrogen indices
    polar_atom_indices = []
    polar_hydrogen_indices = []
    for atom in rdmol.GetAtoms():
        # 1-1. non-alkyl carbons
        if atom.GetSymbol() == 'C' and str(atom.GetHybridization()) != 'SP3':
            polar_atom_indices.append(atom.GetIdx())
    for atom in rdmol.GetAtoms():
        # 1-2. non-alkyl hydrogens
        if atom.GetSymbol() == 'H':
            for bond in atom.GetBonds():
                atom2 = bond.GetOtherAtom(atom)
                if (atom2.GetSymbol() != 'C' and atom2.GetSymbol() !='H'):
                    polar_atom_indices.append(atom.GetIdx())
                elif atom2.GetSymbol() == 'C' and atom2.GetIdx() in polar_atom_indices:
                    polar_atom_indices.append(atom.GetIdx())
                if atom2.GetSymbol() in ['N', 'O']: # [#1:1]-[#7,#8]
                    polar_hydrogen_indices.append(atom.GetIdx()) 
        # 1-3. heteroatoms
        if (atom.GetSymbol() != 'C' and atom.GetSymbol() !='H'):
            polar_atom_indices.append(atom.GetIdx())

    # 2. symmetry Class
    symmetryClass = list(rdchem.CanonicalRankAtoms(rdmol, breakTies=False))

    return polar_atom_indices, polar_hydrogen_indices, symmetryClass
