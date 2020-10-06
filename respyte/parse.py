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
            self.model_type = 'point_charge'
            self.parameter_type = {'charge', 'connectivity'}
            self.q_core_type = None
            self.alpha0 = None
            self.penalty = {'ptype': 'L1', 'a': 0.001, 'b': 0.1}
            self.procedure  = 1
            self.gridinfo = None 
            # self.gridspace = 0.7 
            # self.normalization = False 
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
        self.mols = mols

        # 3. residue charge setting
        resChargeDict = copy.deepcopy(amberResidueChargeDict)
        if 'residue_charge' in inp:
            #print('Read "residue_charge" setting(user-defined fixed residue charge).')
            for resname, charge in inp['residue_charge'].items():
                assert isinstance(charge, (int, float)), f'charge should be a number. charge: {charge} is given for residue,{resname}'
            residue_charge = inp['residue_charge']
            print(f'  * residue_charge: {residue_charge}')
            resChargeDict.update(residue_charge)
        self.resChargeDict = resChargeDict

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
        self.fixed_atomic_charge = fixed_atomic_charge

        # 5. user specified equivalent atoms 
        if 'equiv_atoms' in inp:
            #print('Read "equiv_atoms" setting(user-defined list of equivalent atoms).')
            for group, group_info in inp['equiv_atoms'].items():                
                assert 'molname' in group_info, f'molname not specified in {group}'
                assert 'atomname' in group_info, f'atomname not specified in {group}'
                assert 'resname' in group_info, f'resname not specified in {group}'
            equiv_atoms_inp = inp['equiv_atoms']
        else:
            equiv_atoms_inp = {}

        newequiv_atoms = []
        for i in equiv_atoms_inp:
            molnames = equiv_atoms_inp[i]['molname']
            if isinstance(molnames, str): 
                molnames = [molnames]
            atomnames = equiv_atoms_inp[i]['atomname']
            if isinstance(atomnames, str): 
                atomnames = [atomnames]
            resnames = equiv_atoms_inp[i]['resname']
            if isinstance(resnames, str):
                resname = [resnames]
            newequiv_atoms.append([molnames, atomnames, resnames])
        equiv_atoms = newequiv_atoms
        print(f'  * equiv_atoms: {equiv_atoms}')
        self.equiv_atoms = equiv_atoms

        # 6. charge model type
        if 'model_type'  in  inp: 
            model_type = inp['model_type']
            print(f'  * model_type: {model_type}') 
        else:
            model_type = 'point_charge'
            print(f'  * model_type: {model_type} (default)') 
        self.model_type = model_type

        # 7. parameter type
        if 'parameter_types'  in  inp: 
            parameter_types = inp['parameter_types']
            print(f'  * parameter_types: {parameter_types}') 
        else:
            parameter_types = {'charge': 'connectivity'}
            print(f'  * parameter_types: {parameter_types} (default)') 
        self.parameter_types = parameter_types

        # 8. q_core_type
        if 'q_core_type'  in  inp: 
            q_core_type = inp['q_core_type']
            print(f'  * q_core_type: {q_core_type}') 
        else:
            q_core_type = None
            print(f'  * q_core_type: {q_core_type} (default)') 
        self.q_core_type = q_core_type

        # 9. alpha0 
        if 'alpha0'  in  inp: 
            alpha0 = inp['alpha0']
            print(f'  * alpha0: {alpha0}') 
        else:
            alpha0 = None
            print(f'  * alpha0: {alpha0} (default)') 
        self.alpha0 = alpha0

        # 10. ptype, a, b, c
        if 'penalty' in  inp:
            penalty = inp['penalty']
            print(f'  * penalty: {penalty}') 
        else:
            penalty = {'ptype': 'L1', 'a': 0.001, 'b': 0.1}
            print(f'  * penalty: {penalty} (default)')
        self.penalty = penalty

        # 11. procedure
        if 'procedure' in inp:
            assert isinstance(inp['procedure'], int)
            procedure = inp['procedure']
            print(f'  * procedure: {procedure}')
        else: 
            procedure = 1
            print(f'  * procedure: {procedure} (default)')
        self.procedure = procedure

        # 12. boundary select
        if 'boundary_select' in inp:
            gridinfo = inp['boundary_select']
            print(f'  * gridinfo: {gridinfo}') 
        else:
            gridinfo = None
            print(f'  * gridinfo: {gridinfo} (default)') 
        self.gridinfo = gridinfo

def main():
    inp = Input()
    inp.readinp(sys.argv[1])

if __name__ == "__main__":
    main()
