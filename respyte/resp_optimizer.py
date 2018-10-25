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

# ForceBalance is a dependency
from forcebalance.molecule import *
from readinp import Input
from molecule_resp import Molecule_HJ, Molecule_OEMol
from engine import *
from writemol2 import *
from resp_points_HJ import *
from select_grid import *

# Global variables
bohr2Ang = 0.52918825 # change unit from bohr to angstrom
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

class Respyte_Optimizer:
    def __init__(self, inp = None, molecule = None):
        self.inp = inp
        self.molecule = molecule

    def addInp(self, inpcls):
        assert isinstance(inpcls, Input)
        self.inp = inpcls

    def addMolecule(self, molcls):
        self.molecule = molcls

    def EspDesignMatrix(self, xyzs, gridxyzs, espval):
        """
        Produces a design matrix A and vector B using esp values to solve for charges q in Aq=B.
        Assumes a single input structure.

        Parameters
        ----------
        xyzs : np.ndarray
            Coordinates of atoms whose charges are being fitted.
            Some of these atoms may be topologically equivalent
            but this function doesn't care about that.

        gridxyzs : np.ndarray
            Coordinates of potential / field grid points

        espval : np.ndarray
            Electrostatic potential values

        Returns
        -------
        np.ndarray, np.ndarray

            A matrix and B vector
        """
        nAtoms = xyzs.shape[0]

        molbohr = np.array(xyzs.copy())
        gridx = np.array([xyz[0] for xyz in gridxyzs])
        gridy = np.array([xyz[1] for xyz in gridxyzs])
        gridz = np.array([xyz[2] for xyz in gridxyzs])
        potv = np.array(espval)
        potvsq = potv*potv
        ssvpot = np.sum( potvsq)
        ssvpotsorted = np.sum( np.sort(potvsq) )
        # list of 1/Rij**2 (atom i and point j)
        invRijSq = []
        invRij = []
        dxij = []
        dyij = []
        dzij = []
        for atom in range(nAtoms):
            idx = atom
            dxi = gridx-molbohr[idx,0]
            dxij.append( dxi)
            dyi = gridy-molbohr[idx,1]
            dyij.append( dyi)
            dzi = gridz-molbohr[idx,2]
            dzij.append( dzi)
            rijsq = dxi*dxi + dyi*dyi +dzi*dzi
            invRiSq = 1.0/rijsq
            invRijSq.append( invRiSq )
            invRij.append( np.sqrt( invRiSq) )
        # build A matrix and B vector
        apot = np.zeros( (nAtoms, nAtoms) )
        bpot = np.zeros( nAtoms)
        for j, invRi in enumerate( invRij):
            bi = potv*invRi
            bpot[j] = np.sum( np.sort(bi) )
            for k in range(0, j):
                sumrijrik = np.sum( np.sort(invRi*invRij[k]) )
                apot[j,k] = sumrijrik
                apot[k,j] = sumrijrik
        for j, invRiSq in enumerate( invRijSq):
            apot[j,j] = np.sum( np.sort(invRiSq) )
        return apot, bpot

    def EfDesignMatrix(self, xyzs, gridxyzs, efval):
        """
        Produces a design matrix A and vector B from electric field values to solve for charges q in Aq=B.
        Assumes a single input structure.

        Parameters
        ----------
        xyzs : np.ndarray
            Coordinates of atoms whose charges are being fitted.
            Some of these atoms may be topologically equivalent
            but this function doesn't care about that.

        gridxyzs : np.ndarray
            Coordinates of potential / field grid points

        efval : np.ndarray
            Electric field values

        Returns
        -------
        np.ndarray, np.ndarray

            A matrix and B vector
        """
        nAtoms = xyzs.shape[0]
        molbohr = np.array(xyzs.copy())
        gridx = np.array([xyz[0] for xyz in gridxyzs])
        gridy = np.array([xyz[1] for xyz in gridxyzs])
        gridz = np.array([xyz[2] for xyz in gridxyzs])
        fldv = np.array(efval)
        fldvsq = fldv*fldv
        ssvfld = np.sum( fldvsq)
        ssvfldsorted = np.sum( np.sort(fldvsq) )
        # list of 1/Rij**2 (atom i and point j)
        invRijSq = []
        invRijCubed = []
        invRij = []
        dxij = []
        dyij = []
        dzij = []
        for atom in range(nAtoms):
            idx = atom
            dxi = gridx-molbohr[idx,0]
            dxij.append( dxi)
            dyi = gridy-molbohr[idx,1]
            dyij.append( dyi)
            dzi = gridz-molbohr[idx,2]
            dzij.append( dzi)
            rijsq = dxi*dxi + dyi*dyi +dzi*dzi
            invRiSq = 1.0/rijsq
            absinvRi = np.sqrt(invRiSq)
            invRijSq.append( invRiSq )
            invRij.append(absinvRi)
            invRijCubed.append(invRiSq*absinvRi)
        """
        Under construction.....................................................
        """
        # build A matrix and B vector using electric field.
        dria = np.zeros((nAtoms,len(efval),3))
        for j in range(nAtoms):
            for i in range(len(efval)):
                dria[j][i] = np.array([dxij[j][i],dyij[j][i],dzij[j][i]])
        apot = np.zeros( (nAtoms, nAtoms) )
        bpot = np.zeros( nAtoms)
        for c, invRcCubed in enumerate( invRijCubed):
            bi = invRcCubed*np.einsum('ij,ij->i',efval,dria[c])
            bpot[c] = np.sum(np.sort(bi))
            for a in range(0,c):
                riaric = np.einsum('ij,ij->i',dria[a],dria[c])
                ai = np.sum( np.sort( invRijCubed[a]*invRcCubed*riaric))
                apot[c,a] = ai
                apot[a,c] = ai
            apot[c,c] = np.sum(np.sort(invRij[c]*invRcCubed))
        return apot, bpot

    def EspEfDesignMatrix(self, xyzs, gridxyzs, espval, efval):
        Aesp, Besp = self.EspDesignMatrix(xyzs, gridxyzs, espval)
        Aef, Bef = self.EfDesignMatrix(xyzs, gridxyzs, efval)

        potv = np.array(espval)
        potvsq = potv*potv
        ssvpot = np.sum( potvsq)
        fldv = np.array(efval)
        fldvsq = fldv*fldv
        ssvfld = np.sum( fldvsq)
        # weight = np.sqrt(ssvpot/ssvfld)
        # print('weight:',ssvpot/ssvfld,', sqrt of weight:',np.sqrt(ssvpot/ssvfld))
        # print('compare As:',len(Aesp), Aesp[3],len(Aef), Aef[3])
        # print('compare Bs:',len(Besp), Besp, len(Bef), Bef)
        weight = ssvpot/ssvfld
        Acomb =( Aesp + Aef*weight)
        Bcomb =( Besp + Bef*weight)
        # print('A after mixin:', Acomb[3])
        # print('B after mixin:', Bcomb)
        # input()

        return Acomb, Bcomb

    def LagrangeChargeConstraint(self, aInp, bInp, chargeinfo):
        """
        Returns A matrix and B vector after applying lagrange charge constraint on 'ApplyToCenter'.
        This function should be called before "condensing" the design matrices using equivalent atoms.

        Parameters
        ----------

        aInp : np.ndarray
            "A" matrix; 2D array with dimension (# of atoms)
        bInp : np.ndarray
            "B" matrix; 1D array with dimension (# of atoms)
        ResidueChargeDict : list
            list of lists containing charge of each unit ([[[indices], netcharge], ...])

        Returns
        -------
        np.ndarray, np.ndarray

        constrained A matrix, constrained B vector
        """

        nAtom = len(aInp)
        lastrows = []
        charges = []
        check = []
        unique_resname = list(set(chargeinfo[i][1] for i,info in enumerate(chargeinfo)))
        # print(unique_resname)
        for lstofidx, resname, netchg in chargeinfo:
            if resname in check:
                pass
            else:
                check.append(resname)
                charges.append(netchg)
                ApplytoCenter = [False]*nAtom
                for idx in lstofidx:
                    ApplytoCenter[idx] = True
                lastrow = []
                for setIt in ApplytoCenter:
                    if setIt:
                        lastrow.append(1.0)
                    else:
                        lastrow.append(0.0)
                for i in range(nAtom, nAtom+len(unique_resname)):  # Need to remove duplicates
                    lastrow.append(0.0)
                lastrows.append(lastrow)
        # print(list(len(lastrows[i]) for i in range(len(lastrows))))
        # print('Here', aInp.shape)
        apot = []
        for i, row in enumerate(aInp):
            newrow = list(row)
            for lastrow in lastrows:
                newrow.append(lastrow[i])
            apot.append(np.array(newrow))
        for lastrow in lastrows:
            apot.append(np.array(lastrow))
        bpot = list(bInp)
        for netchg in charges:
            bpot.append(netchg)
        apot = np.array(apot)
        bpot = np.array(bpot)
        # print(apot.shape)
        # input()
        return apot, bpot

    def getCondensedIndices(self, nAtoms, equivGroupsInp):
        """
        Makes a list that stores original indices of element in a condensed matrix.
        For example, if we have 7 atoms starting from zero and the groups (1,4) and (5,6)
        are equivalent, then it should return [0,1,2,3,1,5,5].-> it should be [0,1,2,3,1,4,4] to make it possible to expand after fitting procedure:( 2018.06.23
        Parameters
        ----------
        nAtoms : int
            Total number of atoms of a molecule

        equivGroups : list
            Each element of equivGroups is a list of topologically equivalent atom indices
            where the index is within the range [0, nAtoms)

        Returns
        -------
        np.ndarray
            The length of the list is nAtoms, and each element points to
            a row / column of the condensed design matrix where rows
            and columns are added together.
        """
        equivgroups = copy.deepcopy(np.array(equivGroupsInp))
        listofindices = np.zeros((nAtoms))
        a = -1
        for index in range(nAtoms):
            if not any (index in equivGroup for equivGroup in equivgroups):
                a += 1
                listofindices[index] = a
            elif index in [equivGroup[0] for equivGroup in equivgroups]:
                a += 1
                listofindices[index] = a
        for equivGroup in equivgroups:
            for idx in equivGroup:
                listofindices[idx] = listofindices[equivGroup[0]]
        return np.array(listofindices,dtype=int)

    def Expandqpot(self, qpot_condensed, listofIndices):
        """
        Expands a condensed q vector to an 'expanded' vector with original size.

        Parameters
        ----------

        qpot_condensed : Np.ndarray
            list of elements which contains atomic numbers in the same order of xyz file.

        listofIndices : list
            list of indices of a condensed design matrix.

        Returns
        -------
        list

        List of an expanded q vector.
        """
        nAtoms = len(listofIndices)
        qpot_expanded = np.zeros((nAtoms))
        for idx, i in enumerate(listofIndices):
            qpot_expanded[idx] = qpot_condensed[i]
        return qpot_expanded

    def espRRMS(self,apot, bpot, qpot, espval):
        """
        Calculate relative root mean square (RRMS) (eq 15 in 1993 JPC paper)

        Parameters
        ----------

        apot : np.ndarray
            "A" matrix; 2D array with dimension (# of atoms)
        bpot : np.ndarray
            "B" matrix; 1D array with dimension (# of atoms)
        qpot : np.ndarray
            "q" matrix; 1D array with dimension (# of atoms)
        espval : np.ndarray
            Electrostatic potential values

        Returns
        -------
        float

            RRMS value
        """
        apot = copy.deepcopy(np.array(apot))
        bpot = copy.deepcopy(np.array(bpot))
        qpot = copy.deepcopy(np.array(qpot))
        sumSq = np.sum(np.dot(espval,espval))
        crossProd = np.sum(np.dot(qpot, bpot))
        modelSumSq = np.dot(qpot,np.dot(apot,qpot))
        chiSq = sumSq - 2*crossProd + modelSumSq
        rrmsval = np.sqrt(chiSq/sumSq)
        return rrmsval

    def efRRMS(self,apot, bpot, qpot, efval):
        """
        Calculate relative root mean square (RRMS) (eq 15 in 1993 JPC paper)

        Parameters
        ----------

        apot : np.ndarray
            "A" matrix; 2D array with dimension (# of atoms)
        bpot : np.ndarray
            "B" matrix; 1D array with dimension (# of atoms)
        qpot : np.ndarray
            "q" matrix; 1D array with dimension (# of atoms)
        espval : np.ndarray
            Electrostatic potential values

        Returns
        -------
        float

            RRMS value
        """
        apot = copy.deepcopy(np.array(apot))
        bpot = copy.deepcopy(np.array(bpot))
        qpot = copy.deepcopy(np.array(qpot))
        # sumSq = np.sum(np.dot(espval,espval))
        fldv = np.array(efval)
        fldvsq = fldv*fldv
        ssvfld = np.sum( fldvsq)
        crossProd = np.sum(np.dot(qpot, bpot))
        modelSumSq = np.dot(qpot,np.dot(apot,qpot))
        chiSq = ssvfld - 2*crossProd + modelSumSq
        rrmsval = np.sqrt(chiSq/ssvfld)
        return rrmsval

##########################################################################################################
###               Functions for avoiding any duplicates in different fitting models                    ###
##########################################################################################################
    def combineMatrices(self):
        loc = 0
        apots = []
        bpots = []
        atomid_comb = []
        elem_comb = []
        Size = 0
        for idx, i in enumerate(self.molecule.nmols):
            size = len(self.molecule.xyzs[loc])
            Size += size
            apot_added = np.zeros((size, size))
            bpot_added = np.zeros((size))
            for xyz, gridxyz, espval, efval  in zip(self.molecule.xyzs[loc:loc+i], self.molecule.gridxyzs[loc:loc+i], self.molecule.espvals[loc:loc+i], self.molecule.efvals[loc:loc+i]):
                if 'matrices' in self.inp.restraintinfo:
                    if self.inp.restraintinfo['matrices'] == ['esp', 'ef']:

                        apot, bpot = self.EspEfDesignMatrix(xyz, gridxyz, espval, efval)
                    elif self.inp.restraintinfo['matrices'] == ['esp']:
                        apot, bpot = self.EspDesignMatrix(xyz, gridxyz, espval)
                    elif self.inp.restraintinfo['matrices'] == ['ef']:
                        apot, bpot = self.EfDesignMatrix(xyz, gridxyz, efval)
                else:
                    apot, bpot = self.EspDesignMatrix(xyz, gridxyz, espval)
                apot_added += apot
                bpot_added += bpot
            apot_added /= i
            bpot_added /= i
            apots.append(apot_added)
            bpots.append(bpot_added)
            for j in self.molecule.atomids[idx]:
                atomid_comb.append(j)
            for j in self.molecule.elems[idx]:
                elem_comb.append(j)
            loc += i

        apot_comb  = np.zeros((Size, Size))
        bpot_comb = np.zeros((Size))

        loc = 0
        for idx, apot in enumerate(apots):
            size = len(apot)
            for i in range(size):
                for j in range(size):
                    apot_comb[loc+i][loc+j] = apot[i][j]
            loc += size
        loc = 0
        for bpot in bpots:
            size = len(bpot)
            for i in range(size):
                bpot_comb[loc+i] = bpot[i]
            loc += size
        # print(apots, apot_comb[0], len(apots[0]),'+',len(apots[1]),'=', len(apot_comb))
        # input()
        return apots, bpots, atomid_comb, elem_comb, apot_comb, bpot_comb

    def combine_chargeinfo(self, listofchargeinfo, nAtoms):
        """
        Change local indices in listofchargeinfo ([[[indices], charge], ...]) into global indices
        to apply Lagrange charge constraint into comb matrices.

        """
        loc = 0
        newlistofchargeinfo = []
        # print(listofchargeinfo)
        # input()
        for idx, chargeinfo in enumerate(listofchargeinfo):
            for indices, resname, charge in chargeinfo:
                newindices = [i+loc for i in indices]
                newchargeinfo = [newindices, resname, charge]
                newlistofchargeinfo.append(newchargeinfo)
            loc += nAtoms[idx]
        # print(newlistofchargeinfo)
        # input()
        return newlistofchargeinfo

    def get_nAtoms(self, nList):
        """
        Return a list of the numbers of atoms on each molecule

        """
        nAtoms = []
        for i in nList:
            nAtoms.append(len(i))
        return nAtoms

    def removeSingleElemList(self, lst):
        """
        Remove lists containing single element from input list.

        """
        needtoremove = []
        for idx, i in enumerate(lst):
            if len(i) < 2:
                needtoremove.append(idx)
        needtoremove.sort(reverse= True)
        for i in needtoremove:
            del lst[i]
        return lst

    def get_equivGroup(self, atomid): # but problem is cant exclusively consider polar (or nonpolar) atoms
        """
        Return list of lists containing indices of equivalent atoms by searching the same atom IDs.

        """
        unique_id = set(atomid)
        equivGroup = [[i for i, v in enumerate(atomid) if v == value] for value in unique_id]
        for i in equivGroup:
            i.sort()
        equivGroup.sort()
        equivGroup = self.removeSingleElemList(equivGroup)
        return equivGroup

    def force_symmetry(self, apotInp, bpotInp, elem, atomid, equivGroupInp):
        """
        Force symmetry on matrices using equivGroupInp provided

        """
        apot_sym = copy.deepcopy(apotInp)
        bpot_sym = copy.deepcopy(bpotInp)

        elem_sym = copy.deepcopy(elem)
        atomid_sym = copy.deepcopy(atomid)

        sym = []
        # print(equivGroupInp, len(apot_sym), apot_sym.shape)
        # input()
        for equivlst in equivGroupInp:
            for idx, i in enumerate(equivlst):
                if idx == 0:
                    a = i
                elif idx > 0:
                    # print(a,i)
                    # print(len(apot_sym[a]), len(apot_sym[i]))
                    apot_sym[a,:] += apot_sym[i,:]
                    apot_sym[:,a] += apot_sym[:,i]
                    bpot_sym[a] += bpot_sym[i]
                    sym.append(i)
        sym = sorted(sym,reverse = True)

        for i in sym:
            apot_sym = np.delete(apot_sym, i, axis = 0)
            apot_sym = np.delete(apot_sym, i, axis = 1)
            bpot_sym = np.delete(bpot_sym, i)
            elem_sym = np.delete(elem_sym, i)
            atomid_sym = np.delete(atomid_sym, i)  # after forcing symmetry ,it makes singular matrix problem:/

        return apot_sym, bpot_sym, elem_sym, atomid_sym

    def apply_set_charge(self, apotInp, bpotInp, atomidInp, atomidinfo, set_charge):
        """
        fix charges listed in set_charge and calculate linear algebra

        """
        fixedatoms = []
        setcharges = []
        for atom in set_charge:
            setcharges.append(setcharge[atom]['charge'])
            val = {'resname': set_charge[atom]['resname'], 'atomname': set_charge[atom]['atomname']}
            for idx, atomid in enumerate(atomidInp):
                if val in atomidinfo[atomid]:
                    fixedatoms.append(idx)
        apot_free = copy.deepcopy(apotInp)
        bpot_free = copy.deepcopy(bpotInp)

        for idx, i in enumerate(fixedatoms):
            bpot_free -=setcharges[idx]*apot_free[:,i]
        fixedatoms_rev = sorted(fixedatoms, reverse = True)
        indices_free = []
        for i in range(len(apotInp)):
            if i in fixedatoms:
                pass
            else:
                indices_free.append(i)
        indices_free.sort()

        for i in fixedatoms_rev:
            apot_free = np.delete(apot_free, i, axis = 0)
            apot_free = np.delete(apot_free, i, axis = 1)
            bpot_free = np.delete(bpot_free, i)

        try:
            qpot_free = sci.linalg.solve(apot_free,bpot_free)
        except np.linalg.LinAlgError as err:
            print('Singular matrix error:/ ')
            qpot_free = sci.linalg.lstsq(apot_free, bpot_free)[0] # wonder if this looks right/.... be back later

        qpot_expand = np.zeros((len(apotInp)))
        for idx, i in enumerate(indices_free):
            qpot_expand[i] = qpot_free[idx]
        for idx, i in enumerate(fixedatoms):
            qpot_expand[i] = setcharges[idx]

        return qpot_expand

    def cut_qpot_comb(self, qpot_expanded, nAtoms):
        """
        return list of qpot, a set of charges for each molecule

        """
        qpots = []
        loc = 0
        for i in nAtoms:
            qpot = qpot_expanded[loc: loc+i]
            loc += i
            qpots.append(qpot)
        return qpots
##########################################################################################################
###                       Charge fit models (Model2, Model3, two-stg-fit)                              ###
##########################################################################################################
    def write_mol2(self, atomname, resid, resname, elem, ftype, xyz, qpot, outfilenm = 'out.mol2'):
        row_format = "{:>10}" * 9
        atoms = []
        for elm in elem:
            atom = list(AtomicMass.keys())[elm -1]
            atoms.append(atom)
        with open(outfilenm, 'w') as f:
            f.write('@<TRIPOS>MOLECULE\n')
            f.write('Mol2 generated by resp_optimizer'+'\n')
            f.write('%4i' % len(elem)+'\n')
            if ftype == 'pdb':
                f.write('PROTEIN\n')
            elif ftype == 'xyz':
                f.write('SMALL\n')
            f.write('RESP_CHARGES\n\n')
            f.write('@<TRIPOS>ATOM\n')
            # print(len(xyz), len(atoms))
            # input()
            for idx, i in enumerate(xyz):
                atomidx = '%s%i' % (atoms[idx], (idx+1))
                f.write(row_format.format(idx+1, atomidx, round(i[0]*bohr2Ang,4),
                        round(i[1]*bohr2Ang,4), round(i[2]*bohr2Ang,4), atomname[idx],
                        resid[idx], resname[idx], round(qpot[idx],4)))
                f.write('\n')

    def Model2qpot(self, weight):
        """
        Solves for q vector in Aq=B (Model 2) with a given restraint weight(a). No restraint on hydrogen atoms.

        Parameters
        ----------
        weight : float
            a, scaled facotr which define sthe asymptotic limits of the strength of the restrant.

        Returns
        -------
        list

            list of extended q vectors
        """
        # Combine A matrices and B vectors into apot_comb, bpot_comb
        apots, bpots, atomid_comb, elem_comb, apot_comb, bpot_comb = self.combineMatrices()

        # THen apply Model 2 restraint  -> newapot_comb
        newapot_comb = copy.deepcopy(np.array(apot_comb))
        N = len(newapot_comb)
        for i in range(N):
            if elem_comb[i] == 1:
                pass
            else:
                newapot_comb[i][i] += weight

        # nAtoms : list of number of atoms in each molecule
        nAtoms = self.get_nAtoms(self.molecule.elems)
        # newlistofchareginfo : charge info with 'global indices'
        newlistofchargeinfo = self.combine_chargeinfo(self.molecule.listofchargeinfo, nAtoms)

        # Lagrange Charge Constraint on comb matrices
        apot_constrained, bpot_constrained = self.LagrangeChargeConstraint( newapot_comb, bpot_comb, newlistofchargeinfo)

        # Force symmetry based on the atomid
        equivGroup = self.get_equivGroup(atomid_comb)
        # print(equivGroup)
        # input()
        apot_sym, bpot_sym, elem_sym, atomid_sym = self.force_symmetry(apot_constrained, bpot_constrained, elem_comb, atomid_comb, equivGroup)

        # consider set_charge
        qpot_sym = self.apply_set_charge(apot_sym, bpot_sym, atomid_sym, self.molecule.atomidinfo, self.inp.set_charge)
        indices_sym = self.getCondensedIndices(len(apot_comb), equivGroup)
        qpot_expanded = self.Expandqpot(qpot_sym, indices_sym)

        # split qpot_Expanded into qpots for each molecule
        qpots = self.cut_qpot_comb(qpot_expanded, nAtoms)

        # write mol2 files with fitted charges.
        writeMol2 = False
        path = os.getcwd()
        writeMol2 = False
        path = os.getcwd()
        if os.path.isdir('%s/resp_output' % path):
            print('resp_output dir already exists!!! will overwrite anyway:/')
            writeMol2 = True
        else:
            writeMol2 = True
            os.mkdir('%s/resp_output' % path)

        print()
        print('-------------------------------------------------------')
        print()
        print('                   Model 2 with a =%.4f' % (weight))
        loc = 0
        for idx, i in enumerate(self.molecule.nmols):
            config = 1
            for xyz,gridxyz, espval, efval  in zip(self.molecule.xyzs[loc: loc+i], self.molecule.gridxyzs[loc: loc+i], self.molecule.espvals[loc: loc+i], self.molecule.efvals[loc: loc+i]):
                # if 'matrices' in self.inp.restraintinfo:
                #     if self.inp.restraintinfo['matrices'] == ['esp', 'ef']:
                #         apot, bpot = self.EspEfDesignMatrix(xyz, gridxyz, espval, efval)
                #     elif self.inp.restraintinfo['matrices'] == ['esp']:
                #         apot, bpot = self.EspDesignMatrix(xyz, gridxyz, espval)
                #     elif self.inp.restraintinfo['matrices'] == ['ef']:
                #         apot, bpot = self.EfDesignMatrix(xyz, gridxyz, efval)
                # else:
                #     apot, bpot = self.EspDesignMatrix(xyz, gridxyz, espval)
                Aef, Bef = self.EfDesignMatrix(xyz, gridxyz, efval)
                apot, bpot = self.EspDesignMatrix(xyz, gridxyz, espval)
                print()
                print('           -=#=-  Molecule %d config. %d -=#=-' % (idx+1, config))
                print()
                row_format = "{:>10}" * (5)
                firstrow = ['no.','atomname','resname','atomid','q(opt)']
                print(row_format.format(*firstrow))
                for index, q in enumerate(qpots[idx]):
                    resname = self.molecule.resnames[idx][index]
                    atomname = self.molecule.atomnames[idx][index]
                    atomid = self.molecule.atomids[idx][index]
                    print(row_format.format(index, atomname, resname, atomid,"%.4f" % q))
                print()
                print(' espRRMS : ', "%.4f" % self.espRRMS(apot, bpot, qpots[idx], espval))
                print(' efRRMS  : ', "%.4f" % self.efRRMS(Aef, Bef, qpots[idx], efval))
                if writeMol2 is True:
                    self.write_mol2(self.molecule.atomnames[idx], self.molecule.resnumbers[idx], self.molecule.resnames[idx],
                                    self.molecule.elems[idx], self.molecule.ftype, xyz, qpots[idx],
                                    outfilenm = '%s/resp_output/mol%d_conf%d.mol2'% (path, idx+1, config))
                config += 1
            loc += i
        return 0

    def Model3Amatrix(self, apotInp, weight, tightness, qpotInp, listofelem):
        """
        Builds Model 3 A matrix from A0. No restraint on hydrogen atoms.

        Parameters
        ----------

        apotInp : np.ndarray
            "A" matrix; 2D array with dimension (# of atoms)
        weight : float
            a, scaled factor which defines the asymptotic limits of the strength of the restraint.
        tightness : float
            b, tightness of the hyperbola around its minimum.
        qpotInp : list
            'q' vector
        listofelem : list
            list of elements whose charges are being fitted

        Returns
        -------
        np.ndarray
            Model 3 A matrix
        """
        newapot = copy.deepcopy(np.array(apotInp))
        N =  len(newapot)
        if N != len(listofelem):
            print('List of elements should have the same size with A0 matrix.')
            return False
        for i in range(N):
            if listofelem[i] == 'H' or listofelem[i] == 1:
                continue
            else:
                newapot[i][i] += weight / np.sqrt(qpotInp[i]**2 + tightness**2)
        return newapot

    def Model3qpotFn(self, nAtoms, apot_comb, bpot_comb, weight, tightness, elem_comb, atomid_comb, atomidinfo, chargeinfo_comb, set_charge, equivGroupInp):



        def Model3Iteration(qpot_temp):
            newapot = self.Model3Amatrix(apot_comb, weight, tightness, qpot_temp, elem_comb)
            apot_constrained, bpot_constrained = self.LagrangeChargeConstraint(newapot, bpot_comb, chargeinfo_comb)
            # Force symmetry based on the atomid
            #equivGroup = self.get_equivGroup(atomid_comb)
            apot_sym, bpot_sym, elem_sym, atomid_sym = self.force_symmetry(apot_constrained, bpot_constrained, elem_comb, atomid_comb, equivGroupInp)
            # consider set_charge
            qpot_sym = self.apply_set_charge(apot_sym, bpot_sym, atomid_sym, atomidinfo, set_charge)
            indices_sym = self.getCondensedIndices(len(apot_comb), equivGroupInp)
            qpot_expanded = self.Expandqpot(qpot_sym, indices_sym)
            return qpot_expanded

        Size = len(apot_comb)
        qpotInitial = np.zeros((Size))
        for i in range(10):
            qpotNext = Model3Iteration(qpotInitial)
            if np.linalg.norm(qpotNext-qpotInitial) < 1e-8: break
            qpotInitial = qpotNext.copy()

        # split qpot_Expanded into qpots for each molecule
        qpots = self.cut_qpot_comb(qpotNext, nAtoms)
        return qpots

    def Model3qpot(self, weight, tightness):
        apots, bpots, atomid_comb, elem_comb, apot_comb, bpot_comb = self.combineMatrices()
        nAtoms = self.get_nAtoms(self.molecule.elems)
        equivGroup = self.get_equivGroup(atomid_comb)
        newlistofchargeinfo = self.combine_chargeinfo(self.molecule.listofchargeinfo, nAtoms)
        qpots = self.Model3qpotFn(nAtoms, apot_comb, bpot_comb, weight, tightness,
                                  elem_comb, atomid_comb, self.molecule.atomidinfo,
                                  newlistofchargeinfo, self.inp.set_charge, equivGroup)


        # write mol2 files with fitted charges.
        writeMol2 = False
        path = os.getcwd()
        if os.path.isdir('%s/resp_output' % path):
            print('resp_output dir already exists!!! will overwrite anyway:/')
            writeMol2 = True
        else:
            writeMol2 = True
            os.mkdir('%s/resp_output' % path)

        print()
        print('-------------------------------------------------------')
        print()
        print('              Model 3 with a = %.4f' % (weight))
        loc = 0
        for idx, i in enumerate(self.molecule.nmols):
            config = 1
            for xyz,gridxyz, espval, efval  in zip(self.molecule.xyzs[loc: loc+i], self.molecule.gridxyzs[loc: loc+i], self.molecule.espvals[loc: loc+i], self.molecule.efvals[loc: loc+i]):
                # if 'matrices' in self.inp.restraintinfo:
                #     if self.inp.restraintinfo['matrices'] == ['esp', 'ef']:
                #         apot, bpot = self.EspEfDesignMatrix(xyz, gridxyz, espval, efval)
                #     elif self.inp.restraintinfo['matrices'] == ['esp']:
                #         apot, bpot = self.EspDesignMatrix(xyz, gridxyz, espval)
                #     elif self.inp.restraintinfo['matrices'] == ['ef']:
                #         apot, bpot = self.EfDesignMatrix(xyz, gridxyz, efval)
                # else:
                Aef, Bef = self.EfDesignMatrix(xyz, gridxyz, efval)
                apot, bpot = self.EspDesignMatrix(xyz, gridxyz, espval)
                print()
                print('           -=#=-  Molecule %d config. %d -=#=-' % (idx+1, config))
                print()
                row_format = "{:>10}" * (5)
                firstrow = ['no.','atomname','resname','atomid','q(opt)']
                print(row_format.format(*firstrow))
                for index, q in enumerate(qpots[idx]):
                    resname = self.molecule.resnames[idx][index]
                    atomname = self.molecule.atomnames[idx][index]
                    atomid = self.molecule.atomids[idx][index]
                    print(row_format.format(index, atomname, resname, atomid,"%.4f" % q))
                print()
                print(' espRRMS : ', "%.4f" % self.espRRMS(apot, bpot, qpots[idx], espval))
                print(' efRRMS  : ', "%.4f" % self.efRRMS(Aef, Bef, qpots[idx], efval))
                if writeMol2 is True:
                    self.write_mol2(self.molecule.atomnames[idx], self.molecule.resnumbers[idx], self.molecule.resnames[idx],
                                    self.molecule.elems[idx], self.molecule.ftype, xyz, qpots[idx],
                                    outfilenm = '%s/resp_output/mol%d_conf%d.mol2'% (path, idx+1, config))
                config += 1
            loc += i
        return 0

    def twoStageFit(self, weight1, weight2, tightness):

        apots, bpots, atomid_comb, elem_comb, apot_comb, bpot_comb = self.combineMatrices()
        nAtoms = self.get_nAtoms(self.molecule.elems)
        equivGroup_comb = self.get_equivGroup(atomid_comb) # equivGroup_combined

        # I dont think this 'nonpolars' is that useful. But for now, I need it.
        nonpolars = []
        for index, listofpolar in enumerate(self.molecule.listofpolars):
            nonpolar = []
            for i in range(nAtoms[index]):
                if i in listofpolar:
                    pass
                else:
                    nonpolar.append(i)
            nonpolars.append(nonpolar)
        # print(nonpolars)
        listofpolar_comb = []
        listofnonpolar_comb = []
        loc = 0
        for idx, listofpolar in enumerate(self.molecule.listofpolars):
            for polaridx in listofpolar:
                listofpolar_comb.append(loc+polaridx)
            loc += len(self.molecule.elems[idx])
        for i in range(len(elem_comb)):
            if i not in listofpolar_comb:
                listofnonpolar_comb.append(i)

        listofPolarEquiv = []
        listofNonpolarEquiv = []
        for equivGroup in equivGroup_comb:
            if equivGroup[0] in listofpolar_comb:
                listofPolarEquiv.append(equivGroup)
            else:
                listofNonpolarEquiv.append(equivGroup)

        """
        1st stage: force symmetry on polar atoms and apply smaller restraint weight.

        """
        newlistofchargeinfo = self.combine_chargeinfo(self.molecule.listofchargeinfo, nAtoms)
        qpots_stg1 = self.Model3qpotFn(nAtoms, apot_comb, bpot_comb, weight1, tightness,
                                       elem_comb, atomid_comb, self.molecule.atomidinfo,
                                       newlistofchargeinfo, self.inp.set_charge, listofPolarEquiv)
        """
        2nd stage: force symmetry on nonpolar atoms and while fixing fitted polar charge from 1st stg.

        """
        apot_nonpolar = copy.deepcopy(apot_comb)
        bpot_nonpolar = copy.deepcopy(bpot_comb)
        qpot_nonpolar = []
        for qpot in qpots_stg1:
            for charge in qpot:
                qpot_nonpolar.append(charge)

        # cal. nonpolar charge
        nonpolarchargeinfo = []
        for idx, chargeinfo in enumerate(newlistofchargeinfo):
            indices, resname, charge = chargeinfo
            polarcharges = 0
            nonpolarindices = []
            for index, i in enumerate(indices):
                if i in listofpolar_comb:
                    polarcharges += qpot_nonpolar[i]
                else:
                    nonpolarindices.append(listofnonpolar_comb.index(i))
            nonpolarchargeinfo.append([nonpolarindices,resname, charge-polarcharges])
        # print(newlistofchargeinfo, nonpolarchargeinfo)
        # input()

        atomid_nonpolar = [atomid_comb[i] for i in listofnonpolar_comb]
        elem_nonpolar = [elem_comb[i] for i in listofnonpolar_comb]

        newlistofNonpolarEquiv  = []
        # print('#########################################################')
        # print('listofNonpolarEquiv', listofNonpolarEquiv)
        # print('listofnonpolar_comb', listofnonpolar_comb)
        for nonpolarequivgroup in listofNonpolarEquiv:
            newindices = []
            for i in nonpolarequivgroup:
                idx = listofnonpolar_comb.index(i)
                newindices.append(idx)
            newlistofNonpolarEquiv.append(newindices)
        # print('newlistofNonpolarEquiv', newlistofNonpolarEquiv)
        listofpolar_comb.sort()
        for i in listofpolar_comb:
            bpot_nonpolar -= qpot_nonpolar[i]*apot_nonpolar[:,i]
        listofpolar_comb_rev = sorted(listofpolar_comb, reverse = True)
        for i in listofpolar_comb_rev:
            apot_nonpolar = np.delete(apot_nonpolar, i, axis = 0)
            apot_nonpolar = np.delete(apot_nonpolar, i, axis = 1)
            bpot_nonpolar = np.delete(bpot_nonpolar, i)

        Nnonpolars = []
        for idx, Natom in enumerate(nAtoms):
            Nnonpolar = Natom - len(self.molecule.listofpolars[idx])
            Nnonpolars.append(Nnonpolar)
        # print('Nnonpolar', Nnonpolar)
        qpots_stg2 = self.Model3qpotFn(Nnonpolars, apot_nonpolar, bpot_nonpolar, weight2, tightness,
                                  elem_nonpolar, atomid_nonpolar, self.molecule.atomidinfo,
                                  nonpolarchargeinfo, self.inp.set_charge, newlistofNonpolarEquiv)

        qpots = copy.deepcopy(qpots_stg1)
        for idx, qpot in enumerate(qpots):
            for index, i in enumerate(nonpolars[idx]):
                qpot[i] = qpots_stg2[idx][index]
        # write mol2 files with fitted charges.
        writeMol2 = False
        path = os.getcwd()
        if os.path.isdir('%s/resp_output' % path):
            print('resp_output dir already exists!!! will overwrite anyway:/')
            writeMol2 = True
        else:
            writeMol2 = True
            os.mkdir('%s/resp_output' % path)

        print()
        print('-------------------------------------------------------')
        print()
        print('    Two stage fit with a1 = %.4f, a2 = %.4f' % (weight1, weight2))
        loc = 0
        # print('nmols', self.molecule.nmols)
        # print('len(elems)', len(self.molecule.elems))
        # input()
        for idx, i in enumerate(self.molecule.nmols):
            config = 1
            for xyz,gridxyz, espval, efval  in zip(self.molecule.xyzs[loc: loc+i], self.molecule.gridxyzs[loc: loc+i], self.molecule.espvals[loc: loc+i], self.molecule.efvals[loc: loc+i]):
                # if 'matrices' in self.inp.restraintinfo:
                #     if self.inp.restraintinfo['matrices'] == ['esp', 'ef']:
                #         apot, bpot = self.EspEfDesignMatrix(xyz, gridxyz, espval, efval)
                #     elif self.inp.restraintinfo['matrices'] == ['esp']:
                #         apot, bpot = self.EspDesignMatrix(xyz, gridxyz, espval)
                #     elif self.inp.restraintinfo['matrices'] == ['ef']:
                #         apot, bpot = self.EfDesignMatrix(xyz, gridxyz, efval)
                # else:
                #     apot, bpot = self.EspDesignMatrix(xyz, gridxyz, espval)
                Aef, Bef = self.EfDesignMatrix(xyz, gridxyz, efval)
                apot, bpot = self.EspDesignMatrix(xyz, gridxyz, espval)

                print()
                print('           -=#=-  Molecule %d config. %d -=#=-' % (idx+1, config))
                print()
                row_format = "{:>10}" * (5)
                firstrow = ['no.','atomname','resname','atomid','q(opt)']
                print(row_format.format(*firstrow))
                for index, q in enumerate(qpots[idx]):
                    resname = self.molecule.resnames[idx][index]
                    atomname = self.molecule.atomnames[idx][index]
                    atomid = self.molecule.atomids[idx][index]
                    print(row_format.format(index, atomname, resname, atomid,"%.4f" % q))
                print()
                print(' espRRMS : ', "%.4f" % self.espRRMS(apot, bpot, qpots[idx], espval))
                print(' efRRMS  : ', "%.4f" % self.efRRMS(Aef, Bef, qpots[idx], efval))
                if writeMol2 is True:
                    self.write_mol2(self.molecule.atomnames[idx], self.molecule.resnumbers[idx], self.molecule.resnames[idx],
                                    self.molecule.elems[idx], self.molecule.ftype, xyz, qpots[idx],
                                    outfilenm = '%s/resp_output/mol%d_conf%d.mol2'% (path, idx+1, config))
                config += 1
            loc += i
        return 0

def main():

    inp = Input()
    inp.readinput('input/respyte.yml')

    if inp.cheminformatics =='openeye':
        molecule = Molecule_OEMol()
    else:
        molecule = Molecule_HJ()
    molecule.addInp(inp)
    """
    Under construction!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    """
    cwd = os.getcwd() # the path where resp_optimizer.py is sitting

    # need to gen 'selectedPts'- ingredients: inp.gridinfo['shell_select'], allPts
    listoflistofselectedPts = []

    if inp.grid_gen is True and inp.cheminformatics is None:
        print('Grid_gen is only available when openeye is allowed to be imported')
        raise NotImplementedError
    elif inp.grid_gen is True and inp.cheminformatics == 'openeye':
        print('  Generating grid points.')
        gridType  = inp.gridinfo['type']

        if 'radii' in inp.gridinfo:
            radii = inp.gridinfo['radii']

        for idx, i in enumerate(inp.nmols):
            molN = 'mol%d' % (idx+1)
            wkd = '%s/input/molecules/%s/' %(cwd,molN)
            coordfilepath = []
            listofselectedPts = []
            if i > 1 and os.path.isfile(wkd + '%s.xyz' % (molN)): # In this case, xyz file contains mults conf.
                coordpath = wkd + '%s.xyz' % (molN)
                ftype = 'xyz'
                coordfilepath.append(coordpath)
                raise NotImplementedError
            else:
                for j in range(i):
                    selectedPts = []
                    confN = 'conf%d' % (j+1)
                    path = wkd + '%s/' % (confN)

                    # if os.path.isfile('%sgrid_esp.dat' % path):
                    #     pass
                    # else:
                    for fnm in os.listdir(path):
                        if fnm == '%s_%s.xyz' % (molN, confN):
                            coordpath = path + '%s_%s.xyz' % (molN, confN)
                            ftype = 'xyz'
                            coordfilepath.append(coordpath)
                        elif  fnm == '%s_%s.pdb' % (molN, confN):
                            coordpath = path + '%s_%s.pdb' % (molN, confN)
                            ftype = 'pdb'
                            coordfilepath.append(coordpath)
                        else:
                            continue
                        #############################################################
                        if inp.cheminformatics =='openeye':
                            tempmol = Molecule_OEMol()
                        else:
                            tempmol = Molecule_HJ()
                        tempmol.addInp(inp)
                        if ftype is 'xyz':
                            tempmol.addXyzFiles(coordpath)
                        elif ftype is 'pdb':
                            tempmol.addPdbFiles(coordpath)
                        #############################################################
                        PdbtoMol2(coordpath)
                        molFile = path + '%s_%s.mol2' % (molN, confN)
                        mol = ReadOEMolFromFile(molFile)

                        if radii=='bondi': # way to remove this part? ???
                            oechem.OEAssignRadii(mol, oechem.OERadiiType_BondiVdw)
                        elif radii=='modbondi':
                            oechem.OEAssignRadii(mol, oechem.OERadiiType_BondiHVdw)
                        else:
                            oechem.OEThrow.Fatal('unrecognized radii type %s' % radii)

                        gridOptions = {}
                        gridOptions['space'] = 0.7
                        gridOptions['inner'] = 1.4
                        gridOptions['outer'] = 2.0
                        if gridType == 'extendedmsk' or gridType == 'extendedMsk':
                            gridOptions['gridType'] = 'extendedMsk'
                            gridOptions['space'] = 1.0
                            gridOptions['inner'] = 0.8
                            gridOptions['outer'] = 2.4 # changed
                        elif gridType=='MSK' or gridType=='msk':
                            gridOptions['gridType'] = 'MSK'
                            gridOptions['space'] = 1.0
                        elif gridType == 'newfcc':
                            gridOptions['gridType'] = 'shellFac'
                            gridOptions['inner'] = 0.8
                            gridOptions['outer'] = 2.4 # changed
                        elif gridType=='fcc':
                            gridOptions['gridType'] = 'shellFacConst'
                            gridOptions['outer'] = 1.0
                        elif gridType=='vdwfactors':
                            gridOptions['gridType'] = 'shellFac'
                        elif gridType=='vdwconstants':
                            gridOptions['gridType'] = 'shellConst'
                            gridOptions['inner'] = 0.4
                            gridOptions['outer'] = 1.0
                        else:
                            oechem.OEThrow.Fatal('unrecognized grid type %s' % gridType)


                        if 'inner' in inp.gridinfo:
                            gridOptions['inner'] = inp.gridinfo['inner']
                        if 'outer' in inp.gridinfo:
                            gridOptions['outer'] = inp.gridinfo['outer']
                        if 'space' in inp.gridinfo:
                            gridOptions['space'] = inp.gridinfo['space']

                        if gridOptions['gridType']=='MSK' or gridOptions['gridType'] =='extendedMsk': # changed
                            # generate Merz-Singh-Kollman Connolly surfaces at 1.4, 1.6, 1.8, and 2.0 * vdW radii
                            allPts = resp.GenerateMSKShellPts( mol, gridOptions)
                            if 'shell_select' in inp.gridinfo:
                                for k in inp.gridinfo['shell_select']:
                                    N1 = np.sum([len(allPts[i]) for i in range(k)])
                                    N2 = np.sum([len(allPts[i]) for i in range(k+1)])
                                    for pt in range(N1, N2):
                                        selectedPts.append(pt)
                            else:
                                loc = 0
                                for lst in allPts:
                                    for idx, pt in enumerate(lst):
                                        selectedPts.append(loc+idx)
                                    loc += len(lst)
                            print('Total points:', np.sum([len(allPts[i]) for i in range(len(allPts))]))
                        else:
                            # generate a face-centered cubic grid shell around the molecule using gridOptions
                            allPts = [resp.GenerateGridPointSetAroundOEMol(mol, gridOptions)]
                            if 'boundary_select' in inp.gridinfo: # changed
                                inner = inp.gridinfo['boundary_select']['inner']
                                outer = inp.gridinfo['boundary_select']['outer'] ### Under construction!!!!!(3:30pm)
                                if 'radiiType' in inp.gridinfo['boundary_select']:
                                    radiiType = inp.gridinfo['boundary_select']['radiiType']
                                    if radiiType == 'AlvarezRadii':
                                        radiitype = list(AlvarezRadii)
                                    elif radiiType == 'BondiRadii':
                                        radiitype = list(BondiRadiiOechem)
                                else:
                                    radiitype = BondiRadiiOechem
                                    radiiType = 'BondiRadiiOechem'
                                print('Using %s, inner factor = %0.2f, outer factor = %0.2f'  %(radiiType, inner, outer))
                                selectedPts = SelectGridPts(tempmol, float(inner), float(outer), allPts[0], radiitype)

                                # for k in inp.gridinfo['boundary_select']:
                                #     N1 = np.sum([len(allPts[i]) for i in range(k)])
                                #     N2 = np.sum([len(allPts[i]) for i in range(k+1)])
                                #     for pt in range(N1, N2):
                                #         selectedPts.append(pt)
                            else:
                                loc = 0
                                for lst in allPts:
                                    for idx, pt in enumerate(lst):
                                        selectedPts.append(loc+idx)
                                    loc += len(lst)

                            print('Total points:', np.sum([len(allPts[i]) for i in range(len(allPts))]))
                            print('Selected points: ', len(selectedPts))              # selectedPts = list(range(len(allPts)))
                            # print('Total points:', len(allPts))
                        if os.path.isfile('%sgrid_esp.dat' % path) and inp.gridinfo['force_gen'] is 'N':
                            pass
                        else:
                            ofs = open('%sgrid.dat' % path,'w')

                            for pts in allPts: #changed (for msk-based scheme, allPts is a list of lists)
                                for pt in pts:
                                    ofs.write( '{0:10.6f} {1:10.6f} {2:10.6f}\n'.format(pt[0], pt[1], pt[2]) )
                            ofs.close()
                            print('grid.dat is generated in %s' % path)
                            engine = EnginePsi4()
                            ### should change
                            if 'method' in inp.gridinfo:
                                mthd = inp.gridinfo['method']
                            else:
                                mthd = 'hf'
                            if 'basis' in inp.gridinfo:
                                bss = inp.gridinfo['basis']
                            else:
                                bss = '6-31g*'
                            ################################################################
                            chg = 0
                            for indices, resname, charge in tempmol.listofchargeinfo[0]:
                                chg += charge
                            ################################################################


                            engine.write_input(coordpath, method = mthd, basis = bss, charge = chg, job_path = path)
                            engine.espcal(job_path= path)
                            griddat = path + 'grid.dat'
                            espdat = path + 'grid_esp.dat'
                            efdat = path + 'grid_field.dat'
                            espfoutput = path + '%s_%s.espf' % (molN, confN)
                            engine.genespf(griddat, espdat, efdat, espfoutput)
                            # get output files and add them into one single .espf file
                            # I;m on writing this script////// wait a second...
                    # print('checkkk')
                    # print(inp.gridinfo['shell_select'])
                    # print(len(selectedPts))
                    listofselectedPts.append(selectedPts)
            listoflistofselectedPts.append(listofselectedPts)

    """
    Under construction!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    """


    for idx, i in enumerate(inp.nmols): # need to trim a bit more:
        molN = 'mol%d' % (idx+1)
        wkd = '%s/input/molecules/%s/' %(cwd,molN)
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
                    if fnm == '%s_%s.xyz' % (molN, confN):
                        print('Found %s_%s.xyz' % (molN, confN) )
                        coordpath = path + '%s_%s.xyz' % (molN, confN)
                        ftype = 'xyz'
                        espfpath = path + '%s_%s.espf' %(molN, confN)
                        coordfilepath.append(coordpath)
                        espffilepath.append(espfpath)
                    elif fnm == '%s_%s.pdb' % (molN, confN):
                        print('Found %s_%s.pdb' % (molN, confN) )
                        coordpath = path + '%s_%s.pdb' % (molN, confN)
                        ftype = 'pdb'
                        espfpath = path + '%s_%s.espf' %(molN, confN)
                        coordfilepath.append(coordpath)
                        espffilepath.append(espfpath)
                    else:
                        #print('dummy file?',fnm)
                        continue
        if ftype is 'xyz':
            print(coordfilepath)
            molecule.addXyzFiles(*coordfilepath) # so far, the len(coordfilepath) should be 1.
        elif ftype is 'pdb':
            molecule.addPdbFiles(*coordfilepath)
        # print('check')
        # print([len(listofselectedPts[i]) for i in range(5)]); input()
        molecule.addEspf(*espffilepath, selectedPts =listoflistofselectedPts[idx]) # changed
    os.chdir(cwd) ####Let see if it's right...
    print('resp calculation is running on %s' % cwd)
    cal = Respyte_Optimizer()
    cal.addInp(inp)
    cal.addMolecule(molecule)
    if inp.restraintinfo:
        if inp.restraintinfo['penalty'] == 'model2':
            aval = inp.restraintinfo['a']
            cal.Model2qpot(aval)
        elif inp.restraintinfo['penalty'] == 'model3':
            aval = inp.restraintinfo['a']
            bval = inp.restraintinfo['b']
            cal.Model3qpot(aval, bval)
        elif inp.restraintinfo['penalty'] == '2-stg-fit':
            a1val = inp.restraintinfo['a1']
            a2val = inp.restraintinfo['a2']
            bval = inp.restraintinfo['b']
            cal.twoStageFit(a1val, a2val,bval)

    # print()
    # print('    #####################################################')
    # print('    ###               Test calculations               ###')
    # print('    #####################################################')
    # cal.Model2qpot(0.005)
    # cal.Model3qpot(0.0005,0.1)
    # cal.twoStageFit(0.0005,0.001,0.1)

if __name__ == '__main__':
    main()
