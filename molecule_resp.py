"""
Molecule class (just for refreshing)

"""
import os,sys, copy
import math
import pandas as pd
import scipy.stats as stats
import scipy as sci
import numpy as np
import pylab
import re
from collections import OrderedDict, namedtuple, Counter
from warnings import warn
try:
    import openeye.oechem as oechem
except ImportError:
    warn(' The Openeye module cannot be imported. ( Please provide equivGoups and listofpolar manually.)')
# try:
#     from forcebalance.molecule import *
# except ImportError:
#     warn(' The Forcebalance module cannot be imported. (Cannot read PDB files.)')
from forcebalance.molecule import *

from readinp import Input

# Global variables
bohr2Ang = 0.52918825 # change unit from bohr to angstrom
# Set default residue charges for amino acids (AMBER force field)
aminoAcidUnits = ['ACE', 'NME', 'NHE',
                  'ALA', 'ARG', 'ASN', 'ASP', 'ASH', # ASH: ASP protonated
                  'CYS', 'CYM', 'CYX' , # CYM: deprotonated(-1?), CYX: S-S crosslinking
                  'GLU', 'GLH', 'GLN', 'GLY',
                  'HID', 'HIE', 'HIP', # HIP: protonated
                  'ILE', 'LEU', 'LYS', 'LYN', # LYN: neutral
                  'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL']
ResidueCharge = []
for residue in aminoAcidUnits:
    if residue == 'ARG' or residue == 'LYS' or residue == 'HIP':
        ResidueCharge.append(1)
    elif residue == 'ASP' or residue == 'GLU' or residue == 'CYM':
        ResidueCharge.append(-1)
    else:
        ResidueCharge.append(0)
ResidueChargeDict = OrderedDict()
for idx, i in enumerate(aminoAcidUnits):
    ResidueChargeDict[i] = ResidueCharge[idx]

AtomicMass = OrderedDict([('H' , 1.0079), ('He' , 4.0026),
                          ('Li' , 6.941), ('Be' , 9.0122), ('B' , 10.811), ('C' , 12.0107), ('N' , 14.0067), ('O' , 15.9994), ('F' , 18.9984), ('Ne' , 20.1797),
                          ('Na' , 22.9897), ('Mg' , 24.305), ('Al' , 26.9815), ('Si' , 28.0855), ('P' , 30.9738), ('S' , 32.065), ('Cl' , 35.453), ('Ar' , 39.948),
                          ('K' , 39.0983), ('Ca' , 40.078), ('Sc' , 44.9559), ('Ti' , 47.867), ('V' , 50.9415), ('Cr' , 51.9961), ('Mn' , 54.938), ('Fe' , 55.845), ('Co' , 58.9332),
                          ('Ni' , 58.6934), ('Cu' , 63.546), ('Zn' , 65.39), ('Ga' , 69.723), ('Ge' , 72.64), ('As' , 74.9216), ('Se' , 78.96), ('Br' , 79.904), ('Kr' , 83.8),
                          ('Rb' , 85.4678), ('Sr' , 87.62), ('Y' , 88.9059), ('Zr' , 91.224), ('Nb' , 92.9064), ('Mo' , 95.94), ('Tc' , 98), ('Ru' , 101.07), ('Rh' , 102.9055),
                          ('Pd' , 106.42), ('Ag' , 107.8682), ('Cd' , 112.411), ('In' , 114.818), ('Sn' , 118.71), ('Sb' , 121.76), ('Te' , 127.6), ('I' , 126.9045), ('Xe' , 131.293),
                          ('Cs' , 132.9055), ('Ba' , 137.327), ('La' , 138.9055), ('Ce' , 140.116), ('Pr' , 140.9077), ('Nd' , 144.24), ('Pm' , 145), ('Sm' , 150.36),
                          ('Eu' , 151.964), ('Gd' , 157.25), ('Tb' , 158.9253), ('Dy' , 162.5), ('Ho' , 164.9303), ('Er' , 167.259), ('Tm' , 168.9342), ('Yb' , 173.04),
                          ('Lu' , 174.967), ('Hf' , 178.49), ('Ta' , 180.9479), ('W' , 183.84), ('Re' , 186.207), ('Os' , 190.23), ('Ir' , 192.217), ('Pt' , 195.078),
                          ('Au' , 196.9665), ('Hg' , 200.59), ('Tl' , 204.3833), ('Pb' , 207.2), ('Bi' , 208.9804), ('Po' , 209), ('At' , 210), ('Rn' , 222),
                          ('Fr' , 223), ('Ra' , 226), ('Ac' , 227), ('Th' , 232.0381), ('Pa' , 231.0359), ('U' , 238.0289), ('Np' , 237), ('Pu' , 244),
                          ('Am' , 243), ('Cm' , 247), ('Bk' , 247), ('Cf' , 251), ('Es' , 252), ('Fm' , 257), ('Md' , 258), ('No' , 259),
                          ('Lr' , 262), ('Rf' , 261), ('Db' , 262), ('Sg' , 266), ('Bh' , 264), ('Hs' , 277), ('Mt' , 268)])

class Molecule_HJ:
    def __init__(self, gridxyzs = [], espvals = [], efvals = [], prnlev = 0):
        self.inp = None
        self.xyzs = []
        self.nmols = []
        self.elems = []

        self.atomids = []
        self.atomidinfo = {}

        self.resnames = []
        self.atomnames = []

        self.resnumbers = []
        #self.equivGroups = [] # maybe don't need this
        self.listofpolars = []
        self.listofchargeinfo = []

        self.gridxyzs = []
        for gridxyz in gridxyzs:
            self.addGridPoints(gridxyz)
        self.espvals = []
        for espval in espvals:
            self.addEspValues(espval)
        self.efvals = []
        for efval in efvals:
            self.addEfValues(efval)
        self.prnlev = prnlev

    def addXyzCoordinates(self, xyz):
        if not isinstance(xyz, np.ndarray) or len(xyz.shape) != 2 or xyz.shape[1] != 3:
            print("Problem with input:", xyz)
            raise RuntimeError('Please provide each xyz coordinate set as a numpy array with shape (N_atoms, 3)')
        self.xyzs.append(xyz.copy())
        if self.prnlev >= 1:
            print("Added xyz coordinates with shape %s, %i sets total" % (str(xyz.shape), len(self.xyzs)))
    def addGridPoints(self, gridxyz):
        if not isinstance(gridxyz, np.ndarray) or len(gridxyz.shape) != 2 or gridxyz.shape[1] != 3:
            print(len(gridxyz.shape), gridxyz.shape[1])
            print("Problem with input:", gridxyz)
            raise RuntimeError('Please provide each set of grid points as a numpy array with shape (N_points, 3)')
        # LPW: Assume number of grid points can be different for multiple structures.
        self.gridxyzs.append(gridxyz.copy())
        if self.prnlev >= 1:
            print("Added grid points with shape %s, %i sets total" % (str(gridxyz.shape), len(self.gridxyzs)))
    def addEspValues(self, espval):
        if not isinstance(espval, np.ndarray) or len(espval.shape) != 1:
            print("Problem with input:", espval)
            raise RuntimeError('Please provide each set of ESP values as a 1D numpy array')
        self.espvals.append(espval.copy())
        if self.prnlev >= 1:
            print("Added ESP values with shape %s, %i sets total" % (str(espval.shape), len(self.espvals)))
    def addEfValues(self, efval):
        if not isinstance(efval, np.ndarray) or len(efval.shape) != 2 or efval.shape[1] != 3:
            print("Problem with input:", efval)
            raise RuntimeError('Please provide each set of EF values as a nx3 numpy array')
        self.efvals.append(efval.copy())
        if self.prnlev >= 1:
            print("Added EF values with shape %s, %i sets total" % (str(efval.shape), len(self.efvals)))

    def addInp(self, inpcls):
        assert isinstance(inpcls, Input)
        self.inp = inpcls

    def removeSingleElemList(self, lst):
        needtoremove = []
        for idx, i in enumerate(lst):
            if len(i) < 2:
                needtoremove.append(idx)
        needtoremove.sort(reverse= True)
        for i in needtoremove:
            del lst[i]
        return lst

    # def set_equivGroup(self, charge_equal, atomid, atomidinfo):
    #
    #     new_charge_equals = []
    #     for atmnms, resnms in charge_equal:
    #         if len(atmnms) > 1:
    #             pass # we are neglecting the case when atomname is given as a list for now. Be back later
    #         else:
    #             if resnms is "*":
    #                 new_charge_equal = []
    #                 for atmid, info in atomidinfo.items():
    #                     if any(i['atomname'] == atmnms for i in info):
    #                         new_charge_equal.append(atmid)
    #             else:
    #                 if len(resnms) < 2:
    #                     pass
    #                 else:
    #                     new_charge_equal = []
    #                     for resnm in resnms:
    #                         val = {'resname' : resnm, 'atomname' : atmnms}
    #                         for atmid, info in atomidinfo.items():
    #                             if val in info:
    #                                 new_charge_equal.append(int(atmid))
    #             new_charge_equals.append(new_charge_equal)
    #     listofsetequal = []
    #     for charge_equal in new_charge_equals:
    #         setequal = []
    #         for idx, j in enumerate(atomid):
    #             if j in charge_equal:
    #                 setequal.append(idx)
    #         listofsetequal.append(setequal)
    #     unique_atomid = list(set(atomid))
    #     needtoremove = []
    #     for idx, i in enumerate(unique_atomid):
    #         if any(i in lst for lst in new_charge_equals):
    #             needtoremove.append(idx)
    #     needtoremove.sort(reverse  = True)
    #     for idx in needtoremove:
    #         del unique_atomid[idx]
    #
    #     listofsameatoms = []
    #     for i in unique_atomid:
    #         sameatoms = []
    #         for idx, j in enumerate(atomid):
    #             if j == i:
    #                 sameatoms.append(idx)
    #         listofsameatoms.append(sameatoms)
    #
    #     listofsameatoms = self.removeSingleElemList(listofsameatoms)
    #
    #     equivGroup = []
    #     for i in listofsetequal:
    #         equivGroup.append(i)
    #     for i in listofsameatoms:
    #         equivGroup.append(i)
    #
    #     return equivGroup  # considered ID and charge_equal

    def set_listofpolar(self, listofpolar):
        """Manually assign polar atoms"""
        assert isinstance(listofpolar, (list,))
        self.listofpolars.append(listofpolar)

    def set_charge(self, resChargeDict, atomid, atomidinfo, resnumber):
        """
        Output should be like [[[indices], resname, netcharge], ...]
        """
        idxof1statm = [0]
        resname = atomidinfo[atomid[0]][0]['resname']
        resnmof1statm = [resname]
        check = [resnumber[0]]
        for idx, resn in enumerate(resnumber):
            if resn == check[-1]:
                pass
            else:
                check.append(resn)
                idxof1statm.append(idx)
                resnmof1statm.append(atomidinfo[atomid[idx+2]][0]['resname'])
        chargeinfo = []
        idxof1statm.append(len(atomid))
        # print(idxof1statm,resnmof1statm)
        # input()
        for idx, resnm in enumerate(resnmof1statm):
            charge = resChargeDict[resnm]
            lstofidx  =list(range(idxof1statm[idx], idxof1statm[idx+1]))
            chargeinf = [lstofidx, resnm, charge]
            chargeinfo.append(chargeinf)
        return chargeinfo

    def convert_charge_equal(self, charge_equal, atomidinfo):
        """
        Convert charge_equal which assigns equivalent set of atoms into list of equivalent atom ID.

        """
        new_charge_equals = []
        for atmnms, resnms in charge_equal:
            # Case 1, when single or multiple atomnames are set to be equal in any residues.
            if resnms is '*':
                new_charge_equal = [] # store atom ids set to be equivalent.
                for atmnm in atmnms:
                    for atmid, info in self.atomidinfo.items():

                        if any(x['atomname'] == atmnm for x in info):
                            new_charge_equal.append(atmid)

            # Case 2, when single or multiple atomnames in specific residues are set to be equivalent.
            elif len(atmnms) > 1 or len(resnms) > 1:
                new_charge_equal = []
                for i in atmnms:
                    for j in resnms:
                        val = {'resname' : j, 'atomname' : i}
                        for atmid, info in self.atomidinfo.items():
                            if val in info:
                                new_charge_equal.append(atmid)
            else:
                pass
            new_charge_equal = list(set(new_charge_equal))
            new_charge_equals.append(new_charge_equal)
        new_charge_equals = self.removeSingleElemList(new_charge_equals)
        new_charge_equals.sort()
        return new_charge_equals

    def getidxof1statm(self, listofresid, listofresname):
        idxof1statm = [0]
        resnameof1statm = [listofresname[0]]
        check = [listofresid[0]]
        for idx, resid in enumerate(listofresid):
            if resid == check[-1]:
                pass
            else:
                check.append(resid)
                idxof1statm.append(idx)
                resnameof1statm.append(listofresname[idx])
        return idxof1statm, resnameof1statm

    def addXyzFile(self, xyzFile, resname = None):
        xyzs = []
        firstline = True
        firstframe = True
        if len(self.atomidinfo) == 0:
            atmID = 1
        else:
            atmID = max(list(self.atomidinfo.keys()))+1
        # Read xyzFile
        with open(xyzFile) as xyzfile:
            for line in xyzfile:
                ls = line.strip().expandtabs()
                if firstline == True and ls.isdigit() == True:
                    nAtom = int(ls)
                    firstline = False
                    xyz  = []
                    elem = []
                    atomid = []
                    lstofatomname = []
                    lstofresname = []
                elif re.match("[0-9]+( +[-+]?([0-9]*\.)?[0-9]+){4}$" or "[0-9]+( +[-+]?([0-9]*\.)?[0-9]+){3}$", ls):
                    sline = ls.split()
                    xyz.append([float(i) for i in sline[1:4]])
                    if firstframe is True:
                        atomicNum = int(ls[0])
                        elem.append(atomicNum)
                        atomname = list(AtomicMass.keys())[atomicNum-1]
                        if resname is None:
                            number = len(self.elems)+1
                            resname = 'mol%d' %(number)
                        lstofatomname.append(atomname)
                        lstofresname.append(resname)
                        val = { 'resname' : resname, 'atomname' : atomname}
                        self.atomidinfo[atmID] = val
                        atomid.append(atmID)
                        atmID += 1
                elif re.match("^[A-Z][A-Za-z]?( +[-+]?([0-9]*\.)?[0-9]+){3}$", ls):
                    sline = ls.split()
                    xyz.append([float(i) for i in sline[1:4]])
                    if firstframe is True:
                        atomicNum = list(AtomicMass.keys()).index(ls[0])+1
                        elem.append(atomicNum)
                        atomname = ls[0]
                        if resname is None:
                            number = len(self.elems)+1
                            resname = 'mol%d' %(number)
                        lstofatomname.append(atomname)
                        lstofresname.append(resname)
                        val = { 'resname' : resname, 'atomname' : atomname}
                        self.atomidinfo[atmID] = val
                        atomid.append(atmID)
                        atmID += 1
                elif firstline == False and ls.isdigit() == True  and int(ls) == nAtom:
                    firstframe = False
                    if len(xyz) == nAotm:
                        xyzs.append(np.array(xyz)/bohr2Ang)
                        xyz = []
                    else:
                        raise RuntimeError('The xyz file contains some format error.')
                        break
            if len(xyz) == nAtom:
                xyzs.append(np.array(xyz)/bohr2Ang)
            else:
                raise RuntimeError('The xyz file contains some format error.')
        print('%d conformers have been read from %s.' % (len(xyzs),xyzFile))

        if self.inp is not None:
            equiv_ids = self.convert_charge_equal(self.inp.charge_equal, self.atomidinfo)
        else:
            equiv_ids = []
        newatomid = atomid.copy()
        newnewatomidinfo = self.atomidinfo.copy() # self.atomidinfo or just atomidinfo?
        for equiv_id in equiv_ids:
            newid = equiv_id[0]
            for i in equiv_id[1:]:
                newatomid = [newid if x == i else x for x in newatomid]
                for j in self.atomidinfo[i]:
                    newnewatomidinfo[newid].append(j) # need to remove old one
                del newnewatomidinfo[i]

        self.atomids.append(newatomid)
        self.atomidinfo = newnewatomidinfo
        self.elems.append(elem)
        resnumber = list(np.ones((len(elem))))
        self.resnumbers.append(resnumber)
        self.resnames.append(lstofresname)
        self.atomnames.append(lstofatomname)
        self.nmols.append(len(xyzs))
        for xyz in xyzs:
            self.xyzs.append(xyz)
        resnumber= list(np.ones((len(xyz))))
        indices = list(range(len(elem)))
        if self.inp is not None:
            charge = self.inp.resChargeDict[resname]
        else:
            charge = None
        chargeinfo = [[indices, resname, charge]]
        self.listofchargeinfo.append(chargeinfo)

    def addPdbFiles(self, *pdbFiles):
        xyzs = []
        firstconf = True
        for pdbFile in pdbFiles:
            fbmol = Molecule(pdbFile)
            xyz = fbmol.xyzs[0]
            xyz = np.array(xyz)/bohr2Ang
            xyzs.append(xyz)
            elem = fbmol.elem
            resnumber = fbmol.resid
            resname = fbmol.resname
            atomname = fbmol.atomname
            if firstconf is True:
                firstconf = False
                newelem = []
                for elm in elem:
                    atomicNum =   list(AtomicMass.keys()).index(elm) + 1
                    newelem.append(atomicNum)
                atomid = []

                if len(self.atomidinfo) == 0:
                    atmid = 1
                else:
                    atmid = max(list(self.atomidinfo.keys()))

                for res, atom in zip(resname, atomname):
                    val = {'resname' : res, 'atomname' : atom}
                    #print(list(self.atomidinfo.values()))
                    #input()
                    if len(self.atomidinfo) == 0:
                        #print('Here?')
                        atomid.append(atmid)
                        self.atomidinfo[atmid] = [val]
                        atmid += 1
                        #print(list(self.atomidinfo.values()))
                    elif any(val in v for v in list(self.atomidinfo.values())): # doesnt work...
                        for k ,v in self.atomidinfo.items():
                            if val in v:
                                atomid.append(int(k))
                    else:
                        atomid.append(atmid)
                        self.atomidinfo[atmid] = [val]
                        atmid += 1

            # Using 'charge_equal' in input sett
            if self.inp is not None:
                equiv_ids = self.convert_charge_equal(self.inp.charge_equal, self.atomidinfo)
            else:
                equiv_ids = []
            # and modify atomid and atomidinfo so that they(equivalent atoms) can have the same id?
            # No cant do this. maybe not. can one key possess more than one value?
            newatomid = atomid.copy()
            newnewatomidinfo = self.atomidinfo.copy()
            for equiv_id in equiv_ids:
                newid = equiv_id[0]
                for i in equiv_id[1:]:
                    newatomid = [newid if x == i else x for x in newatomid]
                    for j in self.atomidinfo[i]:
                        newnewatomidinfo[newid].append(j)
                    del newnewatomidinfo[i]

            self.atomids.append(newatomid)
            self.atomidinfo = newnewatomidinfo
            self.elems.append(newelem)
            self.resnames.append(resname)
            self.atomnames.append(atomname)
            self.resnumbers.append(resnumber)
        self.nmols.append(len(xyzs))
        for xyz in xyzs:
            self.xyzs.append(xyz)
        if self.inp is not None:
            chargeinfo = self.set_charge(self.inp.resChargeDict, newatomid, self.atomidinfo, resnumber)
        else:
            indices = list(range(len(elem)))
            charge = None
            number = len(self.elems)+1
            resname = 'mol%d' %(number)
            chargeinfo = [[indices, resname, charge]]
        self.listofchargeinfo.append(chargeinfo)

    def addEspf(self, *espfFiles):
        for espfFile in espfFiles:
            espval = []
            gridxyz = []
            efval = []
            with open(espfFile,'r') as espffs:
                for line in espffs:
                    fields = line.strip().split()
                    numbers = [float(field) for field in fields]
                    if (len(numbers)==4):
                        xyz = [x/bohr2Ang for x in numbers[0:3]]
                        gridxyz.append(xyz)
                        espval.append(numbers[3])
                    elif (len(numbers)==3):
                        efval.append(numbers[0:3])
                    else:
                        print('Error ReadEspfFile: encountered line not having 3 or 4 numbers')
                        return False
            if self.prnlev >= 1:
                print()
                print('ReadEspfFile: % d ESP and % d EF points read in from file %s' % (len(espval), len(efval), espfFile))
                print()
            gridxyz = np.array(gridxyz)
            espval = np.array(espval)
            efval = np.array(efval)
            self.addGridPoints(gridxyz)
            self.addEspValues(espval)
            self.addEfValues(efval)

class Molecule_OEMol(Molecule_HJ):
    def addXyzFile(self, xyzFile,resname = None):
    #     # need to consider the case when user wants to pass multiple conformers as separated xyz files.
    #     if len(xyzFiles) > 1:
    #         # add files into one??
    #         for xyzFile in xyzFiles:
        firstline = True
        startofstructure = []
        with open(xyzFile) as xyzfile:
            for linenum, line in enumerate(xyzfile):
                ls = line.strip().expandtabs()
                if firstline == True and ls.isdigit() == True:
                    nAtom = int(ls)
                    firstline = False
                    startofstructure.append(linenum)
                elif firstline == False and ls.isdigit() == True and int(ls) == nAtom:
                    startofstructure.append(linenum)
        startofstructure.append(len(open(xyzFile).readlines()))
        listofoemol = []
        xyzs = []
        for i in range(len(startofstructure)-1):
            with open(xyzFile) as xyzfile:
                with open('tmp.xyz', 'w') as tmp:
                    for linenum, line in enumerate(xyzfile):
                        if linenum >= startofstructure[i] and linenum < startofstructure[i+1]:
                            tmp.write(line)
            ifs = oechem.oemolistream('tmp.xyz')
            oemol = oechem.OEGraphMol()
            oechem.OEReadMolecule(ifs, oemol)
            listofoemol.append(oemol)
            os.remove('tmp.xyz')
            oechem.OEPerceiveSymmetry(oemol)
            maxIdx = oemol.GetMaxAtomIdx()
            molxyz = oechem.OEFloatArray(3*maxIdx)
            oemol.GetCoords(molxyz)
            molxyz = np.array(molxyz)
            coords = np.resize(molxyz, (maxIdx,3))/bohr2Ang
            xyzs.append(coords)
            if i == 0: # first frame
                elem = []
                symmetryClass = []
                listofpolar = []
                atomid = []
                lstofresname = []
                lstofatomname = []
                if len(self.atomidinfo) == 0:
                    atmID = 1
                else:
                    atmID = max(list(self.atomidinfo.keys()))+1
                for atom in oemol.GetAtoms():
                    elem.append(atom.GetAtomicNum())
                    symmetryClass.append(atom.GetSymmetryClass())
                    atomname = list(AtomicMass.keys())[atom.GetAtomicNum()-1]
                    if resname is None:
                        number = len(self.elems)+1
                        resname = 'mol%d' %(number)
                    lstofresname.append(resname)
                    lstofatomname.append(atomname)

                    val = {'resname' : resname, 'atomname' : atomname}
                    self.atomidinfo[atmID] = [val]
                    atomid.append(atmID)
                    atmID += 1
                    if atom.IsAromatic() or atom.IsPolar() is True:
                        listofpolar.append(atom.GetIdx())
                        for atom2 in oemol.GetAtoms():
                            if atom2.IsHydrogen() and atom2.IsConnected(atom) is True:
                                listofpolar.append(atom2.GetIdx())
                            elif atom2.IsCarbon() and atom2.IsConnected(atom) and oechem.OEGetHybridization(atom2) < 3:
                                listofpolar.append(atom2.GetIdx())
                    listofpolar = sorted(set(listofpolar))

                unique_sym = set(symmetryClass)
                equiv_sym = [[i for i, v in enumerate(symmetryClass) if v == value] for value in unique_sym]
                equiv_sym = self.removeSingleElemList(equiv_sym)

                equiv_ids = []
                for i in equiv_sym:
                    newlst = list(set(atomid[j] for j in i))
                    equiv_ids.append(newlst)

                if self.inp is not None:
                    new_charge_equals = self.convert_charge_equal(self.inp.charge_equal, self.atomidinfo)
                else:
                    new_charge_equals = []
                # equiv_id_comb
                equiv_ids_comb = []
                for i in equiv_ids:
                    equiv_ids_comb.append(i)
                for i in new_charge_equals:
                    equiv_ids_comb.append(i)
                for i in equiv_ids_comb:
                    i.sort()
                equiv_ids_comb.sort()

                newequiv_ids_comb = []
                needtoskip = []
                for idx, equivgrp in enumerate(equiv_ids_comb):

                    if idx in needtoskip:
                        pass
                    else:
                        newequivgrp = equivgrp
                        for idxx, equivgrp2 in enumerate(equiv_ids_comb[idx+1:]):

                            if any(i in equivgrp2 for i in equivgrp ):
                                needtoskip.append(idxx+idx+1)
                                newequivgrp = list(set(newequivgrp + equivgrp2))
                        newequiv_ids_comb.append(newequivgrp)

                equiv_ids_comb = newequiv_ids_comb

                newatomid = atomid.copy()
                newnewatomidinfo = self.atomidinfo.copy()
                for equiv_id in equiv_ids_comb:
                    newid = equiv_id[0]
                    for i in equiv_id[1:]:
                        newatomid = [newid if x == i else x for x in newatomid]
                        for j in self.atomidinfo[i]:
                            newnewatomidinfo[newid].append(j) # need to remove old one
                        del newnewatomidinfo[i]

                self.atomids.append(newatomid)
                self.atomidinfo = newnewatomidinfo
                self.elems.append(elem)
                self.listofpolars.append(listofpolar)
                self.resnames.append(lstofresname)
                self.atomnames.append(lstofatomname)
        print('%d conformers have been read from %s.' % (len(xyzs),xyzFile))
        for xyz in xyzs:
            self.xyzs.append(xyz)
        self.nmols.append(len(xyzs))
        resnumber = list(np.ones((len(elem))))
        self.resnumbers.append(resnumber)
        indices = list(range(len(elem)))
        if self.inp is not None:
            charge = self.inp.resChargeDict[resname]
        else:
            charge = None
        chargeinfo = [[indices, resname, charge]]
        self.listofchargeinfo.append(chargeinfo)

    def addPdbFiles(self, *pdbFiles):
        xyzs = []
        listofoemol = []
        firstconf = True
        for pdbFile in pdbFiles:
            #print(pdbFile)
            fbmol = Molecule(pdbFile)
            xyz = fbmol.xyzs[0]
            xyz = np.array(xyz)/bohr2Ang
            xyzs.append(xyz)
            elem = fbmol.elem # maybe need to change into atomic numbers
            resnumber = fbmol.resid
            resname = fbmol.resname
            atomname = fbmol.atomname
            # Making oemol using openeye toolkit : for atomID(?), equivGroup and listofpolar
            ifs = oechem.oemolistream(pdbFile)
            oemol = oechem.OEGraphMol()
            oechem.OEReadMolecule(ifs, oemol)
            listofoemol.append(oemol)
            oechem.OEPerceiveSymmetry(oemol)
            if firstconf is True:
                firstconf = False
                newelem = []
                for elm in elem:
                    atomicNum =   list(AtomicMass.keys()).index(elm) + 1
                    newelem.append(atomicNum)
                atomid = []

                if len(self.atomidinfo) == 0:
                    atmid = 1
                else:
                    atmid = max(list(self.atomidinfo.keys()))+1
                for res, atom in zip(resname, atomname):
                    val = {'resname' : res, 'atomname' : atom}
                    #print(list(self.atomidinfo.values()))
                    #input()
                    if len(self.atomidinfo) == 0:
                        #print('Here?')
                        atomid.append(atmid)
                        self.atomidinfo[atmid] = [val]
                        atmid += 1
                        #print(list(self.atomidinfo.values()))
                    elif any(val in v for v in list(self.atomidinfo.values())): # doesnt work...
                        for k ,v in self.atomidinfo.items():
                            if val in v:
                                atomid.append(int(k))
                    else:
                        atomid.append(atmid)
                        self.atomidinfo[atmid] = [val]
                        atmid += 1

                symmetryClass = []
                listofpolar = []
                for atom in oemol.GetAtoms():
                    symmetryClass.append(atom.GetSymmetryClass())
                    if atom.IsAromatic() or atom.IsPolar() is True:
                        listofpolar.append(atom.GetIdx())
                        for atom2 in oemol.GetAtoms():
                            if atom2.IsHydrogen() and atom2.IsConnected(atom) is True:
                                listofpolar.append(atom2.GetIdx())
                            elif atom2.IsCarbon() and atom2.IsConnected(atom) and oechem.OEGetHybridization(atom2) < 3:
                                listofpolar.append(atom2.GetIdx())
                listofpolar = sorted(set(listofpolar))

                idxof1statm, resnameof1statm = self.getidxof1statm(resnumber, resname)
                unique_resid = set(resnameof1statm)
                sameresid = [[i for i, v in enumerate(resnameof1statm) if v == value] for value in unique_resid]
                sameresid.sort()
                #sameresid = self.removeSingleElemList(sameresid)
                idxof1statm.append(len(resnumber))
                equiv_ids = []
                #print('symmetryClass', symmetryClass)
                #print('sameresid', sameresid)
                for equivresidgroup in sameresid:
                    resnum = equivresidgroup[0]
                    listofsym = symmetryClass[idxof1statm[resnum]: idxof1statm[resnum +1]]
                    #print(listofsym)
                    unique_sym = set(listofsym)
                    equiv_sym = [[i+idxof1statm[resnum] for i, v in enumerate(listofsym) if v == value] for value in unique_sym]
                    equiv_sym = self.removeSingleElemList(equiv_sym)
                    #print('equiv_sym', equiv_sym)
                    # change index to ID
                    equiv_ID = []
                    for lst in equiv_sym:
                        newlist = []
                        for item in lst:
                            newlist.append(atomid[item])
                        equiv_ID.append(newlist)
                    for i in equiv_ID:
                        i.sort()
                        equiv_ids.append(i) # weird:\
                needtoremove = []
                for idx, equiv_id in enumerate(equiv_ids):
                    if len(set(equiv_id)) == 1:
                        needtoremove.append(idx)
                needtoremove.sort(reverse = True)
                for i in needtoremove:
                    del equiv_ids[i]

                if self.inp is not None:
                    new_charge_equals = self.convert_charge_equal(self.inp.charge_equal, self.atomidinfo)
                else:
                    new_charge_equals = []
                equiv_ids_comb = []
                for i in equiv_ids:
                    equiv_ids_comb.append(i)
                for i in new_charge_equals:
                    equiv_ids_comb.append(i)
                for i in equiv_ids_comb:
                    i.sort()
                equiv_ids_comb.sort()
                print('equiv_ids_comb', equiv_ids_comb)

                # and modify atomid and atomidinfo so that they(equivalent atoms) can have the same id?
                # No cant do this. maybe not. can one key possess more than one value?
                newatomid = atomid.copy()
                newnewatomidinfo = self.atomidinfo.copy() # self.atomidinfo or just atomidinfo?
                for equiv_id in equiv_ids_comb:
                    newid = equiv_id[0]
                    for i in equiv_id[1:]:
                        newatomid = [newid if x == i else x for x in newatomid]
                        for j in self.atomidinfo[i]:
                            newnewatomidinfo[newid].append(j) # need to remove old one
                        del newnewatomidinfo[i]

                self.atomids.append(newatomid)
                self.atomidinfo = newnewatomidinfo
                self.elems.append(newelem)
                self.resnames.append(resname)
                self.atomnames.append(atomname)
                self.resnumbers.append(resnumber)
                self.listofpolars.append(listofpolar)

        self.nmols.append(len(xyzs))
        for xyz in xyzs:
            self.xyzs.append(xyz)
        if self.inp is not None:
            chargeinfo = self.set_charge(self.inp.resChargeDict, newatomid, self.atomidinfo, resnumber)
        else:
            indices = list(range(len(elem)))
            charge = None
            number = len(self.elems)+1
            resname = 'mol%d' % (number)
            chargeinfo = [[indices, resname, charge]]
        self.listofchargeinfo.append(chargeinfo)
def main():

    inp = Input()
    inp.readinput('input/respyte.yml')

    if inp.cheminformatics =='openeye':
        molecule = Molecule_OEMol()
    else:
        molecule = Molecule_HJ()
    molecule.addInp(inp)
    if inp.grid_gen is False:
        # Add coordinates
        for idx, i in enumerate(inp.nmols): # need to trim a bit more:
            molN = 'mol%d' % (idx+1)
            wkd = 'input/molecules/%s/' %(molN)
            coordfilepath = []
            espffilepath = []
            # if i > 1 and os.path.isfile(wkd + '%s.xyz' % (molN)) is False:
            #     print('Please combine xyz files into one xyz file with %s.xyz as its name.' % (molN))
            #     break
            if i > 1 and os.path.isfile(wkd + '%s.xyz' % (molN)): # In this case, xyz file contains mults conf.
                coordpath = wkd + '%s.xyz' % (molN)
                ftype = 'xyz'
                coordfilepath.append(coordpath)
                for j in range(i):
                    confN = 'conf%d' % (j+1)
                    path = wkd + '%s/' % (confN)
                    espfpath = path + '%s_%s.espf' %(molN, confN)
                    espffilepath.append(espfpath)
            else:
                for j in range(i):
                    confN = 'conf%d' % (j+1)
                    path = wkd + '%s/' % (confN)

                    for fnm in os.listdir(path):
                        if fnm.endswith('.xyz'):
                            coordpath = path + '%s_%s.xyz' % (molN, confN)
                            ftype = 'xyz'
                            espfpath = path + '%s_%s.espf' %(molN, confN)

                        elif fnm.endswith('.pdb'):
                            coordpath = path + '%s_%s.pdb' % (molN, confN)
                            ftype = 'pdb'
                            espfpath = path + '%s_%s.espf' %(molN, confN)
                    coordfilepath.append(coordpath)
                    espffilepath.append(espfpath)
            if ftype is 'xyz':
                print(coordfilepath)
                molecule.addXyzFile(*coordfilepath) # so far, the len(coordfilepath) should be 1.
            elif ftype is 'pdb':
                molecule.addPdbFiles(*coordfilepath)
            molecule.addEspf(*espffilepath)
    else:
        print('grid generation is not implemented yet:/')
        raise NotImplementedError
    print('elem',molecule.elems)
    print('listofchargeinfo', len(molecule.listofchargeinfo), molecule.listofchargeinfo)
    print('len(listofchargeinfo)',len(molecule.listofchargeinfo))
    #print('equivGroups', molecule.equivGroups)
    print('atomids', molecule.atomids)
    print(len(molecule.xyzs),'=?',len(molecule.espvals))
    print('listofpolars', molecule.listofpolars)
    print(len(molecule.resnames))
    #molecule.addPdbFile
if __name__ == "__main__":
    main()
