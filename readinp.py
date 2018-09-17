"""
Input class
"""

import os,sys, copy
import math
import pandas as pd
import scipy.stats as stats
import scipy as sci
import numpy as np
import pylab
import re
import yaml

from collections import OrderedDict, namedtuple, Counter
# For AMBER force field:)
aminoAcidUnits = ['ACE', 'NHE', 'NME', # Capping groups
                  'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'CYX', 'GLU', 'GLN', 'GLY', 'HID', 'HIE', 'HIP',
                  'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL']
ResidueCharge = []

for residue in aminoAcidUnits:
    if residue == 'ARG' or residue == 'LYS' or residue == 'HIP':
        ResidueCharge.append(1)
    elif residue == 'ASP' or residue == 'GLU':
        ResidueCharge.append(-1)
    else:
        ResidueCharge.append(0)

ResidueChargeDict = OrderedDict()
for idx, i in enumerate(aminoAcidUnits):
    ResidueChargeDict[i] = ResidueCharge[idx]

class Input:
    def __init__(self):
        self.cheminformatics = None
        self.set_charge = {}
        self.resChargeDict = {}
        self.charge_equal = {}
        self.nmols = []
        self.restraintinfo = {}
        self.charge = None

    def readinput(self, inputFile):

        inp = yaml.load(open(inputFile))

        """ Read Cheminformatics """
        if 'cheminformatics' in inp:
            cheminformatics = inp['cheminformatics']
        else:
            cheminformatics = None

        """ Read total charge (mainly for small organic molecule)"""
        if 'charge' in inp:
            charge = inp['charge']
            self.charge = charge

        """ Read set_charge for charge freezing """
        if 'set_charge' in inp:
            set_charge= inp['set_charge']
        else:
            set_charge = {}

        """
        From net_set, define resChargeDict.
        For small molecule, resname is like mol1, mol2, ... and atomname is the same with element name:)
        """
        if 'net_set'  in inp:
            net_set = inp['net_set']
        else:
            net_set = {}
        resChargeDict = copy.deepcopy(ResidueChargeDict)
        for i in net_set:
            res = net_set[i]['resname']
            setcharge = net_set[i]['rescharge']
            resChargeDict[res] = setcharge

        """atomnames and residue names whose charges are set to be equal:)"""
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

        # for molecule in inp['molecules']:
        #     molN = molecule
        #     if inp['molecules'][molecule] == None:
        #         a.addPdbFile('input/molecules/%s/%s.pdb' % (molN, molN))
        #         a.addEspf('input/molecules/%s/%s.espf' % (molN, molN))
        #     else:
        #         for config in inp['molecules'][molecule]:
        #             confN = config
        #             a.addPdbFile('input/molecules/%s/%s/%s_%s.pdb' % (molN, confN, molN, confN))
        #             a.addEspf('input/molecules/%s/%s/%s_%s.espf' % (molN, confN, molN, confN))
        nmols = []
        for molecule in inp['molecules']:
            if inp['molecules'][molecule] is None:
                nmol = 1
            else:
                nmol = len(inp ['molecules'][molecule])
            nmols.append(nmol)

        """ Read charge fit model (Model2, Model3, 2-stg-fit)"""
        restraint = inp['restraint']
        if restraint == 'model2':
            aval = inp['a']
            restraintinfo = { 'restraint' : restraint,
                              'a'         : aval}
        if restraint == 'model3':
            aval = inp['a']
            bval = inp['b']
            restraintinfo = { 'restraint' : restraint,
                              'a'         : aval,
                              'b'         : bval}

        if restraint =='2-stg-fit':
            a1val = inp['a1']
            a2val = inp['a2']
            bval = inp['b']
            restraintinfo = { 'restraint' : restraint,
                              'a1'        : a1val,
                              'a2'        : a2val,
                              'b'         : bval}

        self.cheminformatics = cheminformatics
        self.set_charge = set_charge
        self.resChargeDict = resChargeDict
        self.charge_equal = charge_equal
        self.nmols = nmols
        self.restraintinfo = restraintinfo

def main():
    inp = Input()
    inp.readinput('input/respyt.yml')
    print(inp.cheminformatics)
    print(inp.charge_equal)
    print(inp.restraintinfo)
if __name__ == "__main__":
    main()
