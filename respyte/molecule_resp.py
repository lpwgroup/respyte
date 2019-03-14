"""
Molecule class designed for RESP Optimization

"""

import os, sys, copy
import numpy as np
import re
from collections import OrderedDict
from warnings import warn
try:
    import openeye.oechem as oechem
except ImportError:
    warn(' The Openeye module cannot be imported. ( Please provide equivGoups and listofpolar manually.)')
from respyte.molecule import *
from respyte.readinp_resp import Input
from pathlib import Path
from respyte.readmol import *

bohr2Ang = 0.52918825 # change unit from bohr to angstrom

class Molecule_respyte:
    def __init__(self, gridxyzs = [], espvals = [], efvals = [], prnlev = 0):
        # store molecule objects first
        self.mols = []
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

        # self.spaces = []

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
    # def addSpaces(self, space):
    #     self.spaces.append(space)

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

    def set_listofpolar(self, listofpolar):
        """Manually assign polar atoms"""
        assert isinstance(listofpolar, (list,))
        self.listofpolars.append(listofpolar)


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

    def gen_chargeinfo(self, resChargeDict, atomid, atomidinfo, resnumber):
        """
        Output should be like [[[indices], resname, netcharge], ...]
        """
        idxof1statm = [0]
        resname = atomidinfo[atomid[0]][0]['resname'] # I think this is also problematic..
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
        terminalidx = []
        for idx, resnm in enumerate(resnmof1statm):
            if resnm == 'ACE' or resnm == 'NME'or resnm == 'NHE':
                for i in range(idxof1statm[idx], idxof1statm[idx+1]):
                    terminalidx.append(i)
            else:
                charge = resChargeDict[resnm]
                lstofidx  =list(range(idxof1statm[idx], idxof1statm[idx+1]))
                chargeinf = [lstofidx, resnm, charge]
                chargeinfo.append(chargeinf)
        if len(terminalidx) ==0:
            pass
        else:
            terminalchginf = [terminalidx, 'terminal', 0]
            chargeinfo.append(terminalchginf)   #Wonder if it makes sense., ......
        return chargeinfo

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

    def addCoordFiles(self, *coordFiles):
        if len(coordFiles) == 0:
            print('No conformer is given? ')
            self.mols.append(Molecule())
            self.atomids.append([])
            self.elems.append([])
            self.resnames.append([])
            self.atomnames.append([])
            self.resnumbers.append([])
            self.listofpolars.append([])
            xyzs = []
            self.nmols.append(len(xyzs))
            indices = []
            charge = 0
            number = len(self.elems)+1
            resname = 'mol%d' %(number)
            chargeinfo = [[indices, resname, charge]]
            self.listofchargeinfo.append(chargeinfo)
        else:
            mols = []
            xyzs  = []
            firstconf = True
            for coordFile in coordFiles:
                fbmol = Molecule(coordFile)
                self.mols.append(fbmol)
                xyz = fbmol.xyzs[0]
                xyz = np.array(xyz)/bohr2Ang
                xyzs.append(xyz)
                if firstconf is True:
                    firstconf is False
                    atomicNum = []
                    elem = fbmol.elem
                    if 'resid' not in list(fbmol.Data.keys()):
                        print(' Are you using xyz file? will assing resid, resname,atomname for you!')
                        resnumber = [1 for i in elem]
                        resname = list('MOL' for i in elem)
                        atomname = ['%s%d' % (i,idx+1)for idx, i in enumerate(elem) ]
                    else:
                        resnumber = fbmol.resid
                        resname = fbmol.resname
                        atomname = fbmol.atomname
                    for elm in elem:
                        atomicNum.append(list(PeriodicTable.keys()).index(elm) + 1 )
                    atomid = []
                    if len(self.atomidinfo) == 0:
                        atmid = 1
                    else:
                        atmid = max(list(self.atomidinfo.keys()))
                    # if resname is 'MOL', assign resname to be moli
                    if resname == list('MOL' for i in elem):
                        fnm = Path(coordFile).stem
                        newresname = fnm.split('_')[0]
                        print(' Is this coord file generated from esp_generator? The residue name is MOL.\n')
                        print(' It reassigns the residue name to %s not to confuse with other molecules while forcing symmetry.' % newresname)
                        resname = list(newresname for i in elem)
                        num = 1
                        for res, atom in zip(resname, elem):
                            val = {'resname': res, 'atomname':'%s%d' %(atom, num) }
                            atomid.append(atmid)
                            self.atomidinfo[atmid] = [val]
                            atmid += 1
                            num += 1
                    else:
                        for res, atom in zip(resname, atomname):
                            val = {'resname': res, 'atomname': atom}
                            if len(self.atomidinfo) == 0:
                                atomid.append(atmid)
                                self.atomidinfo[atmid] = [val]
                                atmid += 1
                            elif any(val in v for v in list(self.atomidinfo.values())):
                                for k, v in self.atomidinfo.items():
                                    if val in v:
                                        atomid.append(int(k))
                            else:
                                atomid.append(atmid)
                                self.atomidinfo[atmid] = [val]
                                atmid += 1
                    if self.inp is not None:
                        equiv_ids = self.convert_charge_equal(self.inp.charge_equal, self.atomidinfo)
                    else:
                        equiv_ids = []
                    # And modify atomid and atomidinfo so that equivalent atoms can have the same id.
                    newatomid = atomid.copy()
                    newatomidinfo = self.atomidinfo.copy()
                    for equiv_id in equiv_ids:
                        newid = equiv_id[0]
                        for i in equiv_id[1:]:
                            newatomid = [newid if x ==i else x for x in newatomid]
                            for j in self.atomidinfo[i]:
                                newatomidinfo[newid].append(j)
                            del newatomidinfo[i]
                    self.atomids.append(newatomid)
                    self.atomidinfo = newatomidinfo
                    self.elems.append(atomicNum)
                    self.resnames.append(resname)
                    self.atomnames.append(atomname)
                    self.resnumbers.append(resnumber)
            self.nmols.append(len(xyzs))
            for xyz in xyzs:
                self.xyzs.append(xyz)
            if self.inp is not None:
                chargeinfo = self.gen_chargeinfo(self.inp.resChargeDict, newatomid, self.atomidinfo, resnumber)
            else:
                indices = list(range(len(elem)))
                charge = None
                number = len(self.elems)+1
                resname = 'mol%d' %(number)
                chargeinfo = [[indices, resname, charge]]
            self.listofchargeinfo.append(chargeinfo)
            # For now, when cheminformatics is not used, ignore polar atoms
            listofpolar = []
            self.listofpolars.append(listofpolar)

    def addEspf(self, *espfFiles, selectedPts):
        for idx, espfFile in enumerate(espfFiles):
            espval = []
            gridxyz = []
            efval = []
            with open(espfFile, 'r') as espff:
                selectedLines = []
                if selectedPts[idx] ==None:
                    for i in range(len(espff.readlines())):
                        selectedLines.append(int(i))
                else:
                    for i in selectedPts[idx]:
                        selectedLines.append(int(i*2))
                        selectedLines.append(int(i*2+1))
            with open(espfFile, 'r') as espff:
                for i, line in enumerate(espff):
                    if i in selectedLines:
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
            # # read space
            # space = 10.0
            # for i in gridxyz:
            #     for j in gridxyz[i:]:
            #         dr = np.array(j) - np.array(i)
            #     r = np.sqrt(np.dot(dr, dr))
            #     if r == 0:
            #         pass
            #     elif r < space:
            #         space = r

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
            # self.addSpaces(space)

class Molecule_OEMol(Molecule_respyte):
    def addCoordFiles(self, *coordFiles):
        if len(coordFiles) == 0:
            print('Skip this molecule? Empty molecule object will be created since no conformers is provided.')
            self.mols.append(oechem.OEMol())
            self.atomids.append([])
            self.elems.append([])
            self.resnames.append([])
            self.atomnames.append([])
            self.resnumbers.append([])
            self.listofpolars.append([])
            xyzs = []
            self.nmols.append(len(xyzs))
            indices = []
            charge = 0
            number = len(self.elems)
            resname = 'mol%d' %(number)
            chargeinfo = [[indices, resname, charge]]
            self.listofchargeinfo.append(chargeinfo)
        else:
            xyzs  = []
            listofoemol = []
            firstconf = True
            for coordFile in coordFiles:
                fbmol = Molecule(coordFile)
                xyz = fbmol.xyzs[0]
                xyz = np.array(xyz)/bohr2Ang
                xyzs.append(xyz)
                # Making oemol using openeye toolkit : for atomID(?), equivGroup and listofpolar
                ifs = oechem.oemolistream(coordFile)
                oemol = oechem.OEGraphMol()
                oechem.OEReadMolecule(ifs, oemol)
                listofoemol.append(oemol)
                oechem.OEPerceiveSymmetry(oemol)

                if firstconf == True:
                    firstconf = False
                    atomicNum = []
                    elem = fbmol.elem
                    if 'resid' not in list(fbmol.Data.keys()):
                        print(' Are you using xyz file? will assing resid, resname,atomname for you!')
                        resnumber = [1 for i in elem]
                        resname = list('MOL' for i in elem)
                        atomname = ['%s%d' % (i,idx+1)for idx, i in enumerate(elem)]
                    else:
                        resnumber = fbmol.resid
                        resname = fbmol.resname
                        atomname = fbmol.atomname
                    for elm in elem:
                        atomicNum.append(list(PeriodicTable.keys()).index(elm) + 1 )
                    atomid = []
                    if len(self.atomidinfo) == 0:
                        atmid = 1
                    else:
                        atmid = max(list(self.atomidinfo.keys())) +1
                    # if resname is 'MOL', assign resname to be moli
                    if resname == list('MOL' for i in elem):
                        fnm = Path(coordFile).stem
                        newresname = fnm.split('_')[0]
                        print(' Is this file generated from esp_generator? The residue name is MOL, which is a default residue name for small organic molecule.')
                        print(' It reassigns the name to %s not to confuse with other molecules while forcing symmetry.' % newresname)
                        resname = list(newresname for i in elem)
                        num = 1
                        for res, atom in zip(resname, elem):
                            val = {'resname': res, 'atomname':'%s%d' %(atom, num) }
                            atomid.append(atmid)
                            self.atomidinfo[atmid] = [val]
                            atmid += 1
                            num += 1
                    else:
                        for res, atom in zip(resname, atomname):
                            val = {'resname': res, 'atomname': atom}
                            if len(self.atomidinfo) == 0:
                                atomid.append(atmid)
                                self.atomidinfo[atmid] = [val]
                                atmid += 1
                            elif any(val in v for v in list(self.atomidinfo.values())):
                                for k, v in self.atomidinfo.items():
                                    if val in v:
                                        atomid.append(int(k))
                            else:
                                atomid.append(atmid)
                                self.atomidinfo[atmid] = [val]
                                atmid += 1
                    # Using openeye tool, make listofpolar,
                    symmetryClass = []
                    listofpolar = []
                    for atom in oemol.GetAtoms():
                        symmetryClass.append(atom.GetSymmetryClass())
                        if atom.IsAromatic() == True or atom.IsPolar() == True:
                            listofpolar.append(atom.GetIdx())
                            for bond in atom.GetBonds():
                                atom2 = bond.GetNbr(atom)
                                if atom2.IsHydrogen():
                                    listofpolar.append(atom2.GetIdx())
                                elif atom2.IsCarbon() and bond.GetOrder() != 1:
                                    listofpolar.append(atom2.GetIdx())
                            # for atom2 in oemol.GetAtoms():
                            #     if atom2.IsHydrogen() and atom2.IsConnected(atom) is True:
                            #         listofpolar.append(atom2.GetIdx())
                            #     elif atom2.IsCarbon() and atom2.IsConnected(atom) and oechem.OEGetHybridization(atom2) < 3:
                            #         listofpolar.append(atom2.GetIdx())

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

                    newatomid = atomid.copy()
                    newatomidinfo = copy.deepcopy(self.atomidinfo)
                    for equiv_id in equiv_ids_comb:
                        newid = equiv_id[0]
                        for i in equiv_id[1:]:
                            newatomid = [newid if x ==i else x for x in newatomid]
                            for j in self.atomidinfo[i]:
                                newatomidinfo[newid].append(j)
                            del newatomidinfo[i]
                    self.atomids.append(newatomid)
                    self.atomidinfo = newatomidinfo
                    self.elems.append(atomicNum)
                    self.resnames.append(resname)
                    self.atomnames.append(atomname)
                    self.resnumbers.append(resnumber)
                    self.listofpolars.append(listofpolar)

                    if self.inp is not None:
                        chargeinfo = self.gen_chargeinfo(self.inp.resChargeDict, newatomid, self.atomidinfo, resnumber)
                    else:
                        indices = list(range(len(elem)))
                        charge = None
                        number = len(self.elems)+1
                        resname = 'mol%d' %(number)
                        chargeinfo = [[indices, resname, charge]]
                    self.listofchargeinfo.append(chargeinfo)
            self.nmols.append(len(xyzs))
            for xyz in xyzs:
                self.xyzs.append(xyz)
            for oemol in listofoemol:
                self.mols.append(oemol)

class Molecule_RDMol(Molecule_respyte):
    def addCoordFiles(self, *coordFiles):
        #raise NotImplementedError('Will be implemented soon!')
        if len(coordFiles) == 0:
            print('Skip this molecule? Empty molecule object will be created since no conformers is provided.')
            self.mols.append(rdchem.Mol())
            self.atomids.append([])
            self.elems.append([])
            self.resnames.append([])
            self.atomnames.append([])
            self.resnumbers.append([])
            self.listofpolars.append([])
            xyzs = []
            self.nmols.append(len(xyzs))
            indices = []
            charge = 0
            number = len(self.elems)
            resname = 'mol%d' %(number)
            chargeinfo = [[indices, resname, charge]]
            self.listofchargeinfo.append(chargeinfo)
        else:
            xyzs  = []
            listofrdmol = []
            firstconf = True
            for coordFile in coordFiles:
                fbmol = Molecule(coordFile)
                xyz = fbmol.xyzs[0]
                xyz = np.array(xyz)/bohr2Ang
                xyzs.append(xyz)
                # Making rdmol using rdkit
                rdmol = ReadRdMolFromFile(coordFile)
                listofrdmol.append(rdmol)
                ##########################################################################
                ### Below is the same with addCoordFiles in Molecule_OEMol             ###
                if firstconf == True:
                    firstconf = False
                    atomicNum = []
                    elem = fbmol.elem
                    if 'resid' not in list(fbmol.Data.keys()):
                        print(' Are you using xyz file? will assing resid, resname,atomname for you!')
                        resnumber = [1 for i in elem]
                        resname = list('MOL' for i in elem)
                        atomname = ['%s%d' % (i,idx+1)for idx, i in enumerate(elem)]
                    else:
                        resnumber = fbmol.resid
                        resname = fbmol.resname
                        atomname = fbmol.atomname
                    for elm in elem:
                        atomicNum.append(list(PeriodicTable.keys()).index(elm) + 1 )
                    atomid = []
                    if len(self.atomidinfo) == 0:
                        atmid = 1
                    else:
                        atmid = max(list(self.atomidinfo.keys())) +1
                    # if resname is 'MOL', assign resname to be moli
                    if resname == list('MOL' for i in elem):
                        fnm = Path(coordFile).stem
                        newresname = fnm.split('_')[0]
                        print(' Is this file generated from esp_generator? The residue name is MOL, which is a default residue name for small organic molecule.')
                        print(' It reassigns the name to %s not to confuse with other molecules while forcing symmetry.' % newresname)
                        resname = list(newresname for i in elem)
                        num = 1
                        for res, atom in zip(resname, elem):
                            val = {'resname': res, 'atomname':'%s%d' %(atom, num) }
                            atomid.append(atmid)
                            self.atomidinfo[atmid] = [val]
                            atmid += 1
                            num += 1
                    else:
                        for res, atom in zip(resname, atomname):
                            val = {'resname': res, 'atomname': atom}
                            if len(self.atomidinfo) == 0:
                                atomid.append(atmid)
                                self.atomidinfo[atmid] = [val]
                                atmid += 1
                            elif any(val in v for v in list(self.atomidinfo.values())):
                                for k, v in self.atomidinfo.items():
                                    if val in v:
                                        atomid.append(int(k))
                            else:
                                atomid.append(atmid)
                                self.atomidinfo[atmid] = [val]
                                atmid += 1
                ### Above is the same with addCoordFiles in Molecule_OEMol             ###
                ##########################################################################
                    # Get symmetry class from rdkit
                    rdchem.AssignStereochemistry(rdmol, cleanIt=True, force=True, flagPossibleStereoCenters=True)
                    symmetryClass = []
                    for atom in rdmol.GetAtoms():
                        symmetryClass.append(int(atom.GetProp('_CIPRank')))
                    # Get a list of polar atoms from rdkit
                    listofpolar  = []
                    for atom in rdmol.GetAtoms():
                        if (atom.GetSymbol() != 'C' and atom.GetSymbol() !='H') or atom.GetIsAromatic():
                            listofpolar.append(atom.GetIdx())
                            for bond in atom.GetBonds():
                                atom2 = bond.GetOtherAtom(atom)
                                if atom2.GetSymbol() == 'H':
                                    listofpolar.append(atom2.GetIdx())
                                elif atom2.GetSymbol() == 'C' and str(bond.GetBondType()) != 'SINGLE':
                                    listofpolar.append(atom2.GetIdx())
                    listofpolar = sorted(set(listofpolar))
                ##########################################################################
                ### Below is the same with addCoordFiles in Molecule_OEMol             ###
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

                    newatomid = atomid.copy()
                    newatomidinfo = copy.deepcopy(self.atomidinfo)
                    for equiv_id in equiv_ids_comb:
                        newid = equiv_id[0]
                        for i in equiv_id[1:]:
                            newatomid = [newid if x ==i else x for x in newatomid]
                            for j in self.atomidinfo[i]:
                                newatomidinfo[newid].append(j)
                            del newatomidinfo[i]
                    self.atomids.append(newatomid)
                    self.atomidinfo = newatomidinfo
                    self.elems.append(atomicNum)
                    self.resnames.append(resname)
                    self.atomnames.append(atomname)
                    self.resnumbers.append(resnumber)
                    self.listofpolars.append(listofpolar)

                    if self.inp is not None:
                        chargeinfo = self.gen_chargeinfo(self.inp.resChargeDict, newatomid, self.atomidinfo, resnumber)
                    else:
                        indices = list(range(len(elem)))
                        charge = None
                        number = len(self.elems)+1
                        resname = 'mol%d' %(number)
                        chargeinfo = [[indices, resname, charge]]
                    self.listofchargeinfo.append(chargeinfo)
            self.nmols.append(len(xyzs))
            for xyz in xyzs:
                self.xyzs.append(xyz)
            for rdmol in listofrdmol:
                self.mols.append(rdmol)
                ### Above is the same with addCoordFiles in Molecule_OEMol             ###
                ##########################################################################


def main():
    # cwd = current working directory in which input folder exists
    cwd = os.getcwd()
    # read respyte.yml
    inp = Input('%s/input/respyte.yml' % (cwd))
    # Create molecule object
    if inp.cheminformatics == 'openeye':
        molecule = Molecule_OEMol()
    elif inp.cheminformatics == 'rdkit':
        molecule = Molecule_RDMol()
    else:
        molecule = Molecule_respyte()
    molecule.addInp(inp)

    for idx, i in enumerate(inp.nmols):
        molN = 'mol%d' % (idx+1)
        wkd = '%s/input/molecules/%s' % (cwd, molN)
        coordfilepath = []
        espffilepath = []
        for j in range(i):
            confN = 'conf%d' % (j+1)
            path = wkd + '/%s' % (confN)
            pdbfile = path + '/%s_%s.pdb' % (molN, confN)
            mol2file = path + '/%s_%s.mol2' % (molN, confN)
            xyzfile = path + '/%s_%s.xyz' % (molN, confN)
            if os.path.isfile(pdbfile):
                coordpath = pdbfile
                coordfilepath.append(coordpath)
            elif os.path.isfile(mol2file):
                coordpath = mol2file
                coordfilepath.append(coordpath)
            elif os.path.isfile(xyzfile):
                coordpath = xyzfile
                coordfilepath.append(coordpath)
                print('This folder doesn not contain pdb or mol2 file format. ')
            else:
                raise RuntimeError(" Coordinate file should have pdb or mol2 file format! ")

            espfpath = path + '/%s_%s.espf' %(molN, confN)
            if not os.path.isfile(espfpath):
                raise RuntimeError('%s file doesnt exist!!! '% espfpath)
            else:
                espffilepath.append(espfpath)

        molecule.addCoordFiles(*coordfilepath)
        print('-----------------------------------------------------------------------------------------')
        print('             ##   Check molecule information of %s  ' % molN )
        print('-----------------------------------------------------------------------------------------')
        #print('elems:       ', molecule.elems,'\n')
        print('atom ids:    ', molecule.atomids,'\n')
        #print('resnames:    ',molecule.resnames,'\n')
        print('polar atoms: ', molecule.listofpolars[-1],'\n')
        #print('charge info: ',molecule.listofchargeinfo[-1],'\n')
        #print('atomidinfo', molecule.atomidinfo)
        #print('-----------------------------------------------------------------------------------------')
    # print(molecule.nmols, len(molecule.nmols))
    print(len(molecule.elems), len(molecule.atomids), len(molecule.resnames), len(molecule.listofpolars), len(molecule.xyzs))
if __name__ == '__main__':
    main()
