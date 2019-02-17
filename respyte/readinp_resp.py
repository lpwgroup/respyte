import os, sys, copy
import numpy as np
import yaml
from warnings import warn
from collections import OrderedDict, namedtuple, Counter

# Set default residue charges for amino acids (AMBER force field)
amberAminoAcidUnits = ['ALA', 'ARG', 'ASN', 'ASP', 'ASH', # ASH: ASP protonated
                  'CYS', 'CYM', 'CYX' , # CYM: deprotonated, CYX: S-S crosslinking
                  'GLU', 'GLH', 'GLN', 'GLY',
                  'HID', 'HIE', 'HIP', # HIP: protonated
                  'ILE', 'LEU', 'LYS', 'LYN', # LYN: neutral
                  'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL']
residueCharge = []
for residue in amberAminoAcidUnits:
    if residue == 'ARG' or residue == 'LYS' or residue == 'HIP':
        residueCharge.append(1)
    elif residue == 'ASP' or residue == 'GLU' or residue == 'CYM':
        residueCharge.append(-1)
    else:
        residueCharge.append(0)
amberResidueChargeDict = OrderedDict()
for idx, i in enumerate(amberAminoAcidUnits):
    amberResidueChargeDict[i] = residueCharge[idx]

class Input:
    def __init__(self, inputFile=None):
        if inputFile is None:
            self.cheminformatics = None
            self.set_charge = {}
            self.resChargeDict = {}
            self.charge_equal = {}
            self.nmols = []
            self.restraintinfo = {}
            self.gridinfo = None
            self.gridspace = 0.7
        else:
            self.readinp(inputFile)

    def readinp(self, inputFile):
        inp = yaml.load(open(inputFile))
        # Read cheminformatics
        if 'cheminformatics' in inp:
            if inp['cheminformatics'] == 'openeye' or inp['cheminformatics'] == 'rdkit' or inp['cheminformatics'] == 'None':
                cheminformatics = inp['cheminformatics']
            else:
                raise NotImplementedError("%s is not implemented. Please choose openeye, rdkit or None:) " % inp['cheminformatics'])
        else:
            cheminformatics = None

        # Read set_charge for charge freezing
        if 'set_charge' in inp:
            set_charge= inp['set_charge']
        else:
            set_charge = {}

        # From net_set, define resChargeDictself.
        # for small molecule, resname is mol1, mol2, ... and atomname = elem
        resChargeDict = copy.deepcopy(amberResidueChargeDict)
        if 'net_set' in inp:
            net_set = inp['net_set']
            resChargeDict.update(net_set)
        if 'charges' in inp:
            resChargeDict.update(inp['charges'])

        # atomnames and resnames whose charges are set to be equal
        if 'charge_equal' in inp:
            charge_equal_inp = inp['charge_equal']
        else:
            charge_equal_inp = {}

        newcharge_equal = []
        for i in charge_equal_inp:
            atomname = charge_equal_inp[i]['atomname']
            resname = charge_equal_inp[i]['resname']
            newcharge_equal.append([atomname, resname])
        charge_equal = newcharge_equal

        # Check how many molecules and how many conformers are set to be fitted
        nmols = []
        for mol in inp['molecules']:
            nmols.append(int(inp['molecules'][mol]))

        # Read charge fit model (Model2, Model3, 2-stg-fit)
        restraintinfo = inp['restraint']

        gridinfo = {}
        # if 'grid_info' in inp:
        #     gridinfo['type'] = inp['grid_info']['type']
        #     gridinfo['radii'] = inp ['grid_info']['radii']

        # if 'shell_select' in inp:
        #    gridinfo['shell_select'] = inp['shell_select']
        if 'boundary_select' in inp:
            gridinfo['boundary_select'] = inp['boundary_select']

        if 'grid_space' in inp:
            space = inp['grid_space']
        else:
            space = 0.7

        self.cheminformatics = cheminformatics
        self.set_charge      = set_charge
        self.resChargeDict   = resChargeDict
        self.charge_equal    = charge_equal
        self.nmols           = nmols
        self.restraintinfo   = restraintinfo
        self.gridinfo        = gridinfo
        self.gridspace       = space

def main():
    inp = Input()
    inp.readinput(sys.argv[1])

if __name__ == "__main__":
    main()
