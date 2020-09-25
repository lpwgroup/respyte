import os, sys, copy
import numpy as np
import yaml
from warnings import warn
from collections import OrderedDict, namedtuple, Counter

# Set default residue charges for amino acids (Residues in AMBER protein force field)
amberAminoAcidUnits = ['ALA', 'ARG', 'ASN', 'ASP', 'ASH', # ASH: ASP protonated
                       'CYS', 'CYM', 'CYX' , # CYM: deprotonated, CYX: S-S crosslinking
                       'GLU', 'GLH', 'GLN', 'GLY',
                       'HID', 'HIE', 'HIP', # HIP: protonated
                       'ILE', 'LEU', 'LYS', 'LYN', # LYN: neutral
                       'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL']

amberResidueChargeDict = OrderedDict()
for residue in amberAminoAcidUnits:
    if residue == 'ARG' or residue == 'LYS' or residue == 'HIP':
        amberResidueChargeDict[residue] = 1
    elif residue == 'ASP' or residue == 'GLU' or residue == 'CYM':
        amberResidueChargeDict[residue] = -1
    else:
        amberResidueChargeDict[residue] = 0

class Input:
    def __init__(self, inputFile=None):
        if inputFile is None:
            self.cheminformatics = None
            self.fixed_atomic_charge = {}
            self.resChargeDict = {}
            self.equiv_atoms = {}
            self.nmols = []
            self.restraintinfo = {}
            self.gridinfo = None
            self.gridspace = 0.7
            self.normalization = False
            self.symmetry  = True
        else:
            self.readinp(inputFile)

    def readinp(self, inputFile):
        print(f'\n\033[1m Parse input file, {os.path.abspath(inputFile)}: \033[0m')
        inp = yaml.load(open(inputFile), yaml.SafeLoader)

        # cheminformatics
        if 'cheminformatics' in inp:
            if inp['cheminformatics'].lower() in  ['openeye', 'rdkit']:
                cheminformatics = inp['cheminformatics'].lower()
            elif inp['cheminformatics'] == 'None': 
                cheminformatics = None
            else:
                raise NotImplementedError("respyte does not support %s. Please choose openeye, rdkit or None. " % inp['cheminformatics'])
        else:
            cheminformatics = None

        # Read fixed_atomic_charge for charge freezing
        if 'fixed_atomic_charge' in inp:
            #print('Read "fixed_atomic_charge" setting(user-defined fixed atomic charge).')
            for atom, charge_info in inp['fixed_atomic_charge'].items():
                assert 'resname' in charge_info, f'resname not specified in {atom}'
                assert 'atomname' in charge_info, f'atomname not specified in {atom}'
                assert 'charge' in charge_info, f'charge not specified in {atom}'
                assert isinstance( charge_info['charge'], (int, float)), f'charge should be a number. charge: {charge_info["charge"]} is given'
            fixed_atomic_charge= inp['fixed_atomic_charge']
            print(f'  * fixed_atomic_charge: {fixed_atomic_charge}')
        else:
            print(f'  * fixed_atomic_charge: None')
            fixed_atomic_charge = {}

        # From residue_charge, define resChargeDictself.
        # for small molecule, resname is mol1, mol2, ... and atomname = elem
        resChargeDict = copy.deepcopy(amberResidueChargeDict)
        if 'residue_charge' in inp:
            #print('Read "residue_charge" setting(user-defined fixed residue charge).')
            for resname, charge in inp['residue_charge'].items():
                assert isinstance(charge, (int, float)), f'charge should be a number. charge: {charge} is given for residue,{resname}'
            residue_charge = inp['residue_charge']
            print(f'  * residue_charge: {residue_charge}')
            resChargeDict.update(residue_charge)

        if 'charges' in inp:
            #print('Read charges setting(user-defined fixed molecule net charge).')
            for molname, charge in inp['charges'].items():
                assert isinstance(charge, (int, float)), f'charge should be a number. charge: {charge} is given for molecule,{molname}'
            print(f'  * charges: {inp["charges"]}')
            resChargeDict.update(inp['charges'])

        # atomnames and resnames whose charges are set to be equal
        if 'equiv_atoms' in inp:
            #print('Read "equiv_atoms" setting(user-defined list of equivalent atoms).')
            for group, group_info in inp['equiv_atoms'].items():
                assert 'atomname' in group_info, f'atomname not specified in {group}'
                assert 'resname' in group_info, f'resname not specified in {group}'
            equiv_atoms_inp = inp['equiv_atoms']
            print(f'  * equiv_atoms: {equiv_atoms_inp}')
        else:
            print(f'  * equiv_atoms: None')
            equiv_atoms_inp = {}

        newequiv_atoms = []
        for i in equiv_atoms_inp:
            atomname = equiv_atoms_inp[i]['atomname']
            resname = equiv_atoms_inp[i]['resname']
            newequiv_atoms.append([atomname, resname])
        equiv_atoms = newequiv_atoms

        nmols = []
        if 'molecules' in inp: 
            for mol in inp['molecules']:    
                nmols.append(int(inp['molecules'][mol])) 
        else:
            raise KeyError('"molecules" key not provided! number of conformers for each molecule should be provided.')

        if 'restraint' in inp:
            restraintinfo = inp['restraint']
        else:
            raise KeyError('"restraint" key not provided!')

        gridinfo = {}
        if 'boundary_select' in inp:
            gridinfo['boundary_select'] = inp['boundary_select']

        if 'grid_space' in inp:
            space = inp['grid_space']
            print(f'  * grid_space: {space}')
        else:
            print(f'  * grid_space: 0.7 (default)')
            space = 0.7

        if 'normalization' in inp:
            assert isinstance(inp['normalization'], bool), 'value for normalization should be True or False'
            normalization = inp['normalization']
            print(f'  * normalization: {normalization}')
        else:
            print(f'  * normalization: False (default)')
            normalization = False

        if 'symmetry' in inp:
            assert isinstance(inp['symmetry'], bool), 'value for symmetry should be True or False'
            symmetry = inp['symmetry']
            print(f'  * symmetry: {symmetry}')
        else:
            print(f'  * symmetry: True (default)')
            symmetry = True

        self.cheminformatics     = cheminformatics
        self.fixed_atomic_charge = fixed_atomic_charge
        self.resChargeDict       = resChargeDict
        self.equiv_atoms         = equiv_atoms
        self.nmols               = nmols
        self.restraintinfo       = restraintinfo
        self.gridinfo            = gridinfo
        self.gridspace           = space
        self.normalization       = normalization
        self.symmetry            = symmetry
def main():
    inp = Input()
    inp.readinp(sys.argv[1])

if __name__ == "__main__":
    main()
