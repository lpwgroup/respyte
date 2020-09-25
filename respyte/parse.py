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
            self.mols = {}
            self.fixed_atomic_charge = {}
            self.resChargeDict = {}
            self.equiv_atoms = []
            self.model = 'point_charge'
            self.penalty = {'type': 'L1', 'a': 0.001, 'b': 0.1}
            self.procedure  = 1
            self.gridinfo = None 
            # self.gridspace = 0.7 
            # self.normalization = False 
            self.symmetry  = 'all'
        else:
            self.readinp(inputFile) 
    def readinp(self, inputfile):
        print(f'\n\033[1mParsing input file, {os.path.abspath(inputfile)}: \033[0m')
        
        inp = yaml.load(open(inputfile), yaml.SafeLoader)
        self.inp_dir = os.path.dirname(os.path.abspath(inputfile))

        # 2. how to store molecules info (# conformers, molecule name, net charges) 
        """
        inp.mols = dict(
            mol1 : {'nconf' : 10, 'net_charge': 0.0}, 
            mol2 : {'nconf' : 10, 'net_charge': 1.0}, ...)
        """
        mols = OrderedDict()
        if 'molecules' in inp: 
            for mol in inp['molecules']:
                mols[mol] ={'nconf': int(inp['molecules'][mol])}
                if 'charges' in inp:
                    if mol in inp['charges']:
                        net_chg = inp['charges'][mol]
                        assert isinstance(net_chg, (int, float)), f'charge should be a number. charge: {net_chg} is given for molecule,{mol}'
                    else: 
                        net_chg = 0
                else: 
                    net_chg = 0
                mols[mol]['net_charge'] = net_chg

        # 3. residue charge setting
        resChargeDict = copy.deepcopy(amberResidueChargeDict)
        if 'residue_charge' in inp:
            #print('Read "residue_charge" setting(user-defined fixed residue charge).')
            for resname, charge in inp['residue_charge'].items():
                assert isinstance(charge, (int, float)), f'charge should be a number. charge: {charge} is given for residue,{resname}'
            residue_charge = inp['residue_charge']
            print(f'  * residue_charge: {residue_charge}')
            resChargeDict.update(residue_charge)

        # 4. fixed atomic charges
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
            
        # 5. equivalent atoms 
        if 'equiv_atoms' in inp:
            #print('Read "equiv_atoms" setting(user-defined list of equivalent atoms).')
            for group, group_info in inp['equiv_atoms'].items():
                assert 'atomname' in group_info, f'atomname not specified in {group}'
                assert 'resname' in group_info, f'resname not specified in {group}'
            equiv_atoms_inp = inp['equiv_atoms']
        else:
            equiv_atoms_inp = {}

        newequiv_atoms = []
        for i in equiv_atoms_inp:
            atomnames = equiv_atoms_inp[i]['atomname']
            if isinstance(atomnames, str): 
                atomnames = [atomnames]
            resnames = equiv_atoms_inp[i]['resname']
            if isinstance(resnames, str):
                resname = [resnames]
            newequiv_atoms.append([atomnames, resnames])
        equiv_atoms = newequiv_atoms
        print(f'  * equiv_atoms: {equiv_atoms}')
        
        # 6. symmetry
        if 'symmetry' in inp:
            assert inp['symmetry'] in ['all', 'polar', 'nosym'], 'symmetry should be either all, polar or nosym'
            symmetry = inp['symmetry']
            print(f'  * symmetry: {symmetry}')
        else:
            print(f'  * symmetry: all (default)')
            symmetry = 'all'        

        # 7. restraint/ grid selection/ normalization/
        # if 'restraint' in inp:
        #     restraintinfo = inp['restraint']
        # else:
        #     raise KeyError('"restraint" key not provided!')
        if 'model'  in  inp: 
            model = inp['model']
            print(f'  * model: {model}') 
        else:
            model = 'point_charge'
            print(f'  * model: {model} (default)') 

        if 'penalty' in  inp:
            penalty = inp['penalty']
            print(f'  * penalty: {penalty}') 
        else:
            penalty = {'type': 'L1', 'a': 0.001, 'b': 0.1}
            print(f'  * penalty: {penalty} (default)')

        if 'procedure' in inp:
            procedure = inp['procedure']
            print(f'  * procedure: {procedure}')
        else: 
            procedure = 1
            print(f'  * procedure: {procedure} (default)')

        if 'boundary_select' in inp:
            gridinfo = inp['boundary_select']
            print(f'  * gridinfo: {gridinfo}') 
        else:
            gridinfo = None
            print(f'  * gridinfo: {gridinfo} (default)') 
        # if 'grid_space' in inp:
        #     space = inp['grid_space']
        #     print(f'  * grid_space: {space}')
        # else:
        #     print(f'  * grid_space: 0.7 (default)')
        #     space = 0.7

        # if 'normalization' in inp:
        #     assert isinstance(inp['normalization'], bool), 'value for normalization should be True or False'
        #     normalization = inp['normalization']
        #     print(f'  * normalization: {normalization}')
        # else:
        #     print(f'  * normalization: False (default)')
        #     normalization = False

        self.mols                = mols
        self.fixed_atomic_charge = fixed_atomic_charge
        self.resChargeDict       = resChargeDict
        self.equiv_atoms         = equiv_atoms
        self.model               = model
        self.penalty             = penalty  
        self.procedure           = procedure
        self.gridinfo            = gridinfo
        # self.gridspace           = space
        # self.normalization       = normalization
        self.symmetry            = symmetry

def main():
    inp = Input()
    inp.readinp(sys.argv[1])

if __name__ == "__main__":
    main()
