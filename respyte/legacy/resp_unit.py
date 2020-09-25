import os,sys, copy
import scipy as sci
import numpy as np
import re
from collections import OrderedDict, namedtuple, Counter
from warnings import warn
try:
    import rdkit.Chem as rdchem
except ImportError:
    warn(' The rdkit module cannot be imported. ' )
try:
    import openeye.oechem as oechem
except ImportError:
    warn(' The Openeye module cannot be imported. ( Please provide equivGoups and listofpolar manually.)')
from respyte.molecule import *
from respyte.readinp_resp import Input

# Global variables
bohr2Ang = 0.52918825 # change unit from bohr to angstrom

class Respyte_Optimizer:
    def __init__(self, inp = None, molecule = None, normalization = False):
        self.inp = inp
        self.molecule = molecule
        self.normalization = normalization
    def addInp(self, inpcls):
        assert isinstance(inpcls, Input)
        self.inp = inpcls
        self.normalization = inpcls.normalization
    def addMolecule(self, molcls):
        self.molecule = molcls
    def Normalization(self, normalization):
        self.normalization = normalization
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

    def NewEspDesignMatrix(self, xyzs, gridxyzs, espval, efval, objN):
        nAtoms = xyzs.shape[0]
        molbohr = np.array(xyzs.copy())
        gridx = np.array([xyz[0] for xyz in gridxyzs])
        gridy = np.array([xyz[1] for xyz in gridxyzs])
        gridz = np.array([xyz[2] for xyz in gridxyzs])
        potv = np.array(espval)
        potvsq = potv*potv
        ssvpot = np.sum( potvsq)
        ssvpotsorted = np.sum( np.sort(potvsq) )

        abs_ef = []
        squared_ef = []
        for vec in efval:
            absef = np.sqrt(np.dot(np.array(vec), np.array(vec)))
            abs_ef.append(absef)
            squared_ef.append(absef*absef)
        abs_ef = np.array(abs_ef)
        squared_ef = np.array(squared_ef)
        squared_esp = potvsq
        abs_esp = np.sqrt(squared_esp)

        if objN == 'obj2':
            weights = abs_esp
        elif objN == 'obj2_2':
            weights = (abs_esp**2 + 0.01**2 )**(1/2)
        elif objN == 'obj3':
            weights = squared_esp
        elif objN == 'obj3_2':
            weights = (squared_esp**2 + 0.01**2 )**(1/2)
        elif objN == 'obj4':
            weights = abs_ef
        elif objN == 'obj4_2':
            weights = (abs_ef**2 + 0.05**2 )**(1/2)
        elif objN == 'obj5':
            weights = squared_ef
        else:
            raise NotImplementedError('%s is not implemented. Try obj2, obj3, obj4, or obj5 instead!' % objN)
        invRijSq = []
        ainvRijSq = []
        invRij = []
        ainvRij = []
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
            ainvRijSq.append(weights* invRiSq)
            invRij.append( np.sqrt( invRiSq) )
            ainvRij.append(weights*np.sqrt( invRiSq))
        # build A matrix and B vector
        apot = np.zeros( (nAtoms, nAtoms) )
        bpot = np.zeros( nAtoms)
        for j, ainvRi in enumerate( ainvRij):
            bi = potv*ainvRi
            bpot[j] = np.sum( np.sort(bi) )
            for k in range(0, j):
                sumrijrik = np.sum( np.sort(ainvRi*invRij[k]) )
                apot[j,k] = sumrijrik
                apot[k,j] = sumrijrik
        for j, ainvRiSq in enumerate( ainvRijSq):
            apot[j,j] = np.sum(ainvRiSq)

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
        weight = ssvpot/ssvfld
        Acomb =( Aesp + Aef*weight)/2
        Bcomb =( Besp + Bef*weight)/2
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
        for lstofidx, resname, netchg in chargeinfo:
            if len(lstofidx) == 0 :
                unique_resname.remove(resname)
        for lstofidx, resname, netchg in chargeinfo:
            if len(lstofidx) == 0:
                pass
            elif resname in check:
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
        chiSq_rstr = np.sum((qpot**2 + (0.1)**2)**(0.5)-0.1)
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
        fldv = np.array(efval)
        fldvsq = fldv*fldv
        ssvfld = np.sum( fldvsq)
        crossProd = np.sum(np.dot(qpot, bpot))
        modelSumSq = np.dot(qpot,np.dot(apot,qpot))
        chiSq = ssvfld - 2*crossProd + modelSumSq
        rrmsval = np.sqrt(chiSq/ssvfld)
        return rrmsval

    def centerofmass(self, xyzs, elems):
        M = sum(float(list(PeriodicTable.values())[int(elems[i]-1)]) for i in range(len(elems)))
        return np.sum([xyzs[i,:] *float(list(PeriodicTable.values())[int(elems[i]-1)]) / M for i in range(len(xyzs))],axis=0)

    def MMdipole(self, xyzs, elems, qpot):
        com = self.centerofmass(xyzs, elems)
        mux = 0
        muy = 0
        muz = 0
        for qi, xyzi in zip(qpot, xyzs):
            mux += qi*(xyzi[0]-com[0])
            muy += qi*(xyzi[1]-com[1])
            muz += qi*(xyzi[2]-com[2])
            ri_squared = np.dot(np.array(xyzi)-com,np.array(xyzi)-com)
        mu_squared = mux**2+ muy**2+muz**2
        mu =  np.sqrt(mu_squared)*bohr2ang/0.20819434
        return mu

    def Ncond(self, apotInp, elemInp, qpotInp, Options):
        AA = copy.deepcopy(apotInp)
        for i in range(len(elemInp)):
            if elemInp[i] == 'H' or elemInp[i] == 1:
                continue
            else:
                if 'weights' in Options and 'tightness' in Options:
                    weights = Options['weights']
                    AA[i][i] -= weights[i]*qpotInp[i] / np.sqrt(qpotInp[i]**2 + float(Options['tightness'])**2)**3
                else:
                    continue
        try:
            w, v = np.linalg.eig(AA)
            cond = max(abs(w))/min(abs(w))
        except:
            print('condition number calculation failed.')
            cond = 0
        return cond

    def combineMatrices(self):
        loc = 0
        apots = []
        bpots = []
        atomid_comb = []
        elem_comb = []
        Size = 0
        add = 0
        for idx, i in enumerate(self.molecule.nmols):
            if i == 0:
                add += 1
            else:
                size = len(self.molecule.xyzs[loc])
                Size += size
                apot_added = np.zeros((size, size))
                bpot_added = np.zeros((size))
                for mol, xyz, gridxyz, espval, efval  in zip(self.molecule.mols[loc+add:loc+i+add], self.molecule.xyzs[loc:loc+i], self.molecule.gridxyzs[loc:loc+i], self.molecule.espvals[loc:loc+i], self.molecule.efvals[loc:loc+i]):
                    if 'matrices' in self.inp.restraintinfo:
                        if self.inp.restraintinfo['matrices'] == ['esp', 'ef']:
                            apot, bpot = self.EspEfDesignMatrix(xyz, gridxyz, espval, efval)
                        elif self.inp.restraintinfo['matrices'] == ['esp']:
                            apot, bpot = self.EspDesignMatrix(xyz, gridxyz, espval)
                        elif self.inp.restraintinfo['matrices'] == ['ef']:
                            apot, bpot = self.EfDesignMatrix(xyz, gridxyz, efval)
                        elif self.inp.restraintinfo['matrices'] == ['obj2']:
                            apot, bpot = self.NewEspDesignMatrix(xyz, gridxyz, espval, efval, 'obj2')
                        elif self.inp.restraintinfo['matrices'] == ['obj2_2']:
                            apot, bpot = self.NewEspDesignMatrix(xyz, gridxyz, espval, efval, 'obj2_2')
                        elif self.inp.restraintinfo['matrices'] == ['obj3']:
                            apot, bpot = self.NewEspDesignMatrix(xyz, gridxyz, espval, efval, 'obj3')
                        elif self.inp.restraintinfo['matrices'] == ['obj3_2']:
                            apot, bpot = self.NewEspDesignMatrix(xyz, gridxyz, espval, efval, 'obj3_2')
                        elif self.inp.restraintinfo['matrices'] == ['obj4']:
                            apot, bpot = self.NewEspDesignMatrix(xyz, gridxyz, espval, efval, 'obj4')
                        elif self.inp.restraintinfo['matrices'] == ['obj4_2']:
                            apot, bpot = self.NewEspDesignMatrix(xyz, gridxyz, espval, efval, 'obj4_2')
                        elif self.inp.restraintinfo['matrices'] == ['obj5']:
                            apot, bpot = self.NewEspDesignMatrix(xyz, gridxyz, espval, efval, 'obj5')
                    else:
                        apot, bpot = self.EspDesignMatrix(xyz, gridxyz, espval)
                    apot_added += apot
                    bpot_added += bpot
                apot_added /= i
                bpot_added /= i
                if self.normalization == True:
                    norm = sci.linalg.norm(apot_added, 'fro')
                    prefac  = 77.87* len(self.molecule.elems[idx]) / (norm * 6.0)
                    print('Norm:', sci.linalg.norm(apot_added, 'fro'), ', prefac(77.87*Natom/norm/6):', prefac)
                    apot_added *= prefac
                    bpot_added *= prefac

                apots.append(apot_added)
                bpots.append(bpot_added)
                for j in self.molecule.atomids[idx]:
                    atomid_comb.append(j)
                for j in self.molecule.elems[idx]:
                    elem_comb.append(j)
                loc += i
        totN = 0
        for mol in self.molecule.elems:
            totN += len(mol)
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

        return apots, bpots, atomid_comb, elem_comb, apot_comb, bpot_comb

    def combine_chargeinfo(self, listofchargeinfo, nAtoms):
        """
        Change local indices in listofchargeinfo ([[[indices], charge], ...]) into global indices
        to apply Lagrange charge constraint into comb matrices.

        """
        loc = 0
        newlistofchargeinfo = []
        for idx, chargeinfo in enumerate(listofchargeinfo):
            for indices, resname, charge in chargeinfo:
                newindices = [i+loc for i in indices]
                newchargeinfo = [newindices, resname, charge]
                newlistofchargeinfo.append(newchargeinfo)
            loc += nAtoms[idx]
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

    def force_symmetry(self, apotInp, bpotInp, elem, atomid, listofweights, equivGroupInp):
        """
        Force symmetry on matrices using equivGroupInp provided

        """
        apot_sym = copy.deepcopy(apotInp)
        bpot_sym = copy.deepcopy(bpotInp)

        elem_sym = copy.deepcopy(elem)
        atomid_sym = copy.deepcopy(atomid)
        weights_sym = copy.deepcopy(listofweights)
        sym = []
        for equivlst in equivGroupInp:
            for idx, i in enumerate(equivlst):
                if idx == 0:
                    a = i
                elif idx > 0:
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
            weights_sym = np.delete(weights_sym, i)

        return apot_sym, bpot_sym, elem_sym, atomid_sym, weights_sym

    def apply_fixed_atomic_charge(self, apotInp, bpotInp, elemInp, atomidInp, atomidinfo, fixed_atomic_charge, weightsInp, tightness):
        """
        fix charges listed in fixed_atomic_charge and calculate linear algebra

        """
        fixedatoms = []
        setcharges = []
        for atom in fixed_atomic_charge:
            val = {'resname': fixed_atomic_charge[atom]['resname'], 'atomname': fixed_atomic_charge[atom]['atomname']}
            for idx, atomid in enumerate(atomidInp):
                if val in atomidinfo[atomid]:
                    if idx not in fixedatoms:
                        fixedatoms.append(idx)
                        setcharges.append(fixed_atomic_charge[atom]['charge'])
        apot_free = copy.deepcopy(apotInp)
        bpot_free = copy.deepcopy(bpotInp)
        elem_free = copy.deepcopy(elemInp)
        weights_free = copy.deepcopy(weightsInp)

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
            elem_free = np.delete(elem_free, i)
            weights_free = np.delete(weights_free, i)
        if len(apot_free) == 0:
            qpot_free = []
        else:
            try:
                qpot_free = sci.linalg.solve(apot_free,bpot_free)
            except np.linalg.LinAlgError as err:
                print(' Singular matrix error:/ ')
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

    def write_output(self, mol, qpot, moltype='oemol', outfile= 'out.mol2'):
        if moltype is 'oemol':
            for idx, atom in enumerate(mol.GetAtoms()):
                atom.SetPartialCharge(qpot[idx])

            ofs = oechem.oemolostream()
            ofs.SetFormat(oechem.OEFormat_MOL2)
            ofs.open(outfile)
            oechem.OEWriteMolecule(ofs, mol)

        elif moltype is 'rdmol':
            if isinstance(mol, rdchem.Mol):
                print(' Can not generate mol2 file without using Openeye toolkit yet, will be implemented soon!')
            else:
                raise RuntimeError(' Failed to identify the RDKit molecule object!')
        else:
            if isinstance(mol, Molecule):
                print(' Can not generate mol2 file without using Openeye toolkit yet, will be implemented soon!')
            else:
                raise RuntimeError('Failed to identify the molecule object.')

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
        list_of_weights = []
        for i in range(N):
            list_of_weights.append(weight)

        for i in range(N):
            if listofelem[i] == 'H' or listofelem[i] == 1:
                continue
            else:
                newapot[i][i] += weight / np.sqrt(qpotInp[i]**2 + tightness**2)

        return newapot, list_of_weights

    def Model3qpotFn(self, nAtoms, apot_comb, bpot_comb, weight, tightness, elem_comb, atomid_comb, atomidinfo, chargeinfo_comb, fixed_atomic_charge, equivGroupInp):

        def Model3Iteration(qpot_temp):
            newapot, list_of_weights = self.Model3Amatrix(apot_comb, weight, tightness, qpot_temp, elem_comb)
            apot_constrained, bpot_constrained = self.LagrangeChargeConstraint(newapot, bpot_comb, chargeinfo_comb)
            apot_sym, bpot_sym, elem_sym, atomid_sym, weights_sym = self.force_symmetry(apot_constrained, bpot_constrained, elem_comb, atomid_comb, list_of_weights, equivGroupInp)
            qpot_sym = self.apply_fixed_atomic_charge(apot_sym, bpot_sym, elem_sym, atomid_sym, atomidinfo, fixed_atomic_charge, weights_sym, tightness)
            indices_sym = self.getCondensedIndices(len(apot_comb), equivGroupInp)
            qpot_expanded = self.Expandqpot(qpot_sym, indices_sym)
            return qpot_expanded, list_of_weights, apot_sym, bpot_sym, elem_sym, qpot_sym

        Size = len(apot_comb)
        qpotInitial = np.zeros((Size))
        for i in range(50):
            qpotNext, list_of_weights, apot_sym, bpot_sym, elem_sym, qpot_sym = Model3Iteration(qpotInitial)
            if np.linalg.norm(qpotNext-qpotInitial) < 1e-8: break
            qpotInitial = qpotNext.copy()

        #calculate condition number after fitting
        Options = {'weights':list_of_weights, 'tightness': tightness}
        cond1 = self.Ncond( apot_comb, elem_comb, qpotNext, Options)
        cond2 = self.Ncond( apot_sym, elem_sym, qpot_sym, Options)
        # print('###############################################################')
        # print('  After expansion:', cond1)
        # print('  Before expansion:', cond2)
        # print('###############################################################')
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
                                  newlistofchargeinfo, self.inp.fixed_atomic_charge, equivGroup)

        # write mol2 files with fitted charges.
        writeMol2 = False
        path = os.getcwd()
        if os.path.isdir('%s/resp_output' % path):
            print('\033[1;31m resp_output dir already exists!!! will overwrite anyway:/\x1b[0m')            
            writeMol2 = True
        else:
            writeMol2 = True
            os.mkdir('%s/resp_output' % path)
        # write text file for collecting result:)
        with open('%s/resp_output/result.txt' % path,'w') as f:
            f.write('respyte result\n')
            f.write(' Model 3')
            if self.inp.restraintinfo['matrices'] == ['esp', 'ef']:
                f.write(' RESPF\n')
            elif self.inp.restraintinfo['matrices'] == ['esp']:
                f.write(' RESP\n')
            elif self.inp.restraintinfo['matrices'] == ['ef']:
                f.write(' REF\n')
            f.write(' weight = %8.4f, tightness = %8.4f\n' % (weight, tightness))
            for idx, qpot in enumerate(qpots):
                f.write('mol%d\n' % (idx+1))
                for charge in qpot:
                    f.write('%8.4f\n' % round(charge,4))
        print()
        print('-------------------------------------------------------')
        print()
        print('              Model 3 with a = %.4f' % (weight))
        loc = 0
        esprrmss = []
        efrrmss  = []
        add = 0
        for idx, i in enumerate(self.molecule.nmols):
            if i == 0:
                add += 1
                esprrmss.append(0)
                efrrmss.append(0)
            else:
                config = 1
                for xyz,gridxyz, espval, efval  in zip(self.molecule.xyzs[loc: loc+i], self.molecule.gridxyzs[loc: loc+i], self.molecule.espvals[loc: loc+i], self.molecule.efvals[loc: loc+i]):
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
                    esprrms = self.espRRMS(apot, bpot, qpots[idx], espval)
                    efrrms  = self.efRRMS(Aef, Bef, qpots[idx], efval)
                    esprrmss.append(esprrms)
                    efrrmss.append(efrrms)

                    print(' espRRMS : ', "%.4f" % esprrms)
                    print(' efRRMS  : ', "%.4f\n" % efrrms)
                    if writeMol2 is True:
                        if self.inp.cheminformatics == 'openeye':
                            molt = 'oemol'
                        elif self.inp.cheminformatics == 'rdkit':
                            molt = 'rdmol'
                        else:
                            molt = 'fbmol'
                        self.write_output(self.molecule.mols[loc+config-1+ add], qpots[idx], moltype = molt,                                                                                            outfile = '%s/resp_output/mol%d_conf%d.mol2'% (path, idx+1, config))
                    config += 1
            loc += i
        return qpots, esprrmss, efrrmss

    def twoStageFit(self, weight1, weight2, tightness):

        apots, bpots, atomid_comb, elem_comb, apot_comb, bpot_comb= self.combineMatrices()
        nAtoms = self.get_nAtoms(self.molecule.elems)
        equivGroup_comb = self.get_equivGroup(atomid_comb)
        nonpolars = []
        for index, listofpolar in enumerate(self.molecule.listofpolars):
            nonpolar = []
            for i in range(nAtoms[index]):
                if i in listofpolar:
                    pass
                else:
                    nonpolar.append(i)
            nonpolars.append(nonpolar)
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
                                       newlistofchargeinfo, self.inp.fixed_atomic_charge, listofPolarEquiv)

        print()
        print('-------------------------------------------------------')
        print()
        print('     Charges from the first stage (for checking)        ')
        loc = 0
        esprrmss = []
        efrrmss  = []
        add = 0
        for idx, i in enumerate(self.molecule.nmols):
            if i == 0:
                add += 1
                esprrmss.append(0)
                efrrmss.append(0)
            else:
                config = 1
                for xyz,gridxyz, espval, efval  in zip(self.molecule.xyzs[loc: loc+i], self.molecule.gridxyzs[loc: loc+i], self.molecule.espvals[loc: loc+i], self.molecule.efvals[loc: loc+i]):
                    Aef, Bef = self.EfDesignMatrix(xyz, gridxyz, efval)
                    apot, bpot = self.EspDesignMatrix(xyz, gridxyz, espval)

                    print()
                    print('   -=#=- (!!1st stg!!) Molecule %d config. %d -=#=-' % (idx+1, config))
                    print()
                    row_format = "{:>10}" * (5)
                    firstrow = ['no.','atomname','resname','atomid','q(opt)']
                    print(row_format.format(*firstrow))
                    for index, q in enumerate(qpots_stg1[idx]):
                        resname = self.molecule.resnames[idx][index]
                        atomname = self.molecule.atomnames[idx][index]
                        atomid = self.molecule.atomids[idx][index]
                        print(row_format.format(index, atomname, resname, atomid,"%.4f" % q))
                    print()
                    esprrms = self.espRRMS(apot, bpot, qpots_stg1[idx], espval)
                    efrrms  = self.efRRMS(Aef, Bef, qpots_stg1[idx], efval)
                    esprrmss.append(esprrms)
                    efrrmss.append(efrrms)
                    print(' espRRMS(1st stg) : ', "%.4f" % esprrms )
                    print(' efRRMS(1st stg)  : ', "%.4f\n" % efrrms  )
                    config += 1

            loc += i


        """
        2nd stage: force symmetry on nonpolar atoms and while fixing fitted polar charge from 1st stg.

        """
        apot_nonpolar = copy.deepcopy(apot_comb)
        bpot_nonpolar = copy.deepcopy(bpot_comb)
        qpot_nonpolar = []
        for qpot in qpots_stg1:
            for charge in qpot:
                qpot_nonpolar.append(charge)

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
        atomid_nonpolar = [atomid_comb[i] for i in listofnonpolar_comb]
        elem_nonpolar = [elem_comb[i] for i in listofnonpolar_comb]

        newlistofNonpolarEquiv  = []
        for nonpolarequivgroup in listofNonpolarEquiv:
            newindices = []
            for i in nonpolarequivgroup:
                idx = listofnonpolar_comb.index(i)
                newindices.append(idx)
            newlistofNonpolarEquiv.append(newindices)
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
        qpots_stg2 = self.Model3qpotFn(Nnonpolars, apot_nonpolar, bpot_nonpolar, weight2, tightness,
                                  elem_nonpolar, atomid_nonpolar, self.molecule.atomidinfo,
                                  nonpolarchargeinfo, self.inp.fixed_atomic_charge, newlistofNonpolarEquiv)

        qpots = copy.deepcopy(qpots_stg1)
        for idx, qpot in enumerate(qpots):
            for index, i in enumerate(nonpolars[idx]):
                qpot[i] = qpots_stg2[idx][index]
        # write mol2 files with fitted charges.
        writeMol2 = False
        path = os.getcwd()
        if os.path.isdir('%s/resp_output' % path):
            print('\033[1;31m resp_output dir already exists!!! will overwrite anyway:/\x1b[0m')
            writeMol2 = True
        else:
            writeMol2 = True
            os.mkdir('%s/resp_output' % path)
        # write text file for collecting result:)
        with open('%s/resp_output/result.txt' % path,'w') as f:
            f.write('respyte result\n')
            f.write(' Two stage fit')
            if self.inp.restraintinfo['matrices'] == ['esp', 'ef']:
                f.write(' RESPF\n')
            elif self.inp.restraintinfo['matrices'] == ['esp']:
                f.write(' RESP\n')
            elif self.inp.restraintinfo['matrices'] == ['ef']:
                f.write(' REF\n')
            f.write(' weight1 = %8.4f, weight2 = %8.4f, tightness = %8.4f\n' % (weight1, weight2, tightness))
            f.write('cheminformatics: %s\n' % self.inp.cheminformatics)
            for idx, qpot in enumerate(qpots):
                f.write('mol%d\n' % (idx+1))
                for charge in qpot:
                    f.write('%8.4f\n' % round(charge,4))

        print()
        print('-------------------------------------------------------')
        print()
        print('    Two stage fit with a1 = %.4f, a2 = %.4f' % (weight1, weight2))
        loc = 0
        esprrmss = []
        efrrmss  = []
        add = 0
        for idx, i in enumerate(self.molecule.nmols):
            if i == 0:
                add += 1
                esprrmss.append(0)
                efrrmss.append(0)
            else:
                config = 1
                for xyz,gridxyz, espval, efval  in zip(self.molecule.xyzs[loc: loc+i], self.molecule.gridxyzs[loc: loc+i], self.molecule.espvals[loc: loc+i], self.molecule.efvals[loc: loc+i]):
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
                    print(' MM dipole from two-stg-fit: ', self.MMdipole(xyz, self.molecule.elems[idx], qpots[idx]))
                    print()
                    esprrms = self.espRRMS(apot, bpot, qpots[idx], espval)
                    efrrms  = self.efRRMS(Aef, Bef, qpots[idx], efval)
                    esprrmss.append(esprrms)
                    efrrmss.append(efrrms)
                    print(' espRRMS : ', "%.4f" % esprrms)
                    print(' efRRMS  : ', "%.4f\n" % efrrms)
                    
                    if writeMol2 is True:
                        if self.inp.cheminformatics == 'openeye':
                            molt = 'oemol'
                        elif self.inp.cheminformatics == 'rdkit':
                            molt = 'rdmol'
                        else:
                            molt = 'fbmol'
                        self.write_output(self.molecule.mols[loc+config-1+ add], qpots[idx], moltype = molt, outfile = '%s/resp_output/mol%d_conf%d.mol2'% (path, idx+1, config))
                    config += 1

            loc += i

        return  qpots, esprrmss, efrrmss

    def Model4Amatrix(self, apotInp, weight1, weight2, tightness, qpotInp, listofelem,  listofpolar_comb):
        """
        Builds Model 4 A matrix from A0. No restraint on hydrogen atoms.

        Parameters
        ----------

        apotInp : np.ndarray
            "A" matrix; 2D array with dimension (# of atoms)
        weight1 : float
            a1, scaled factor(for nonpolar atoms) which defines the asymptotic limits of the strength of the restraint.
        weight2 : float
            a2, scaled factor(for polar atoms) which defines the asymptotic limits of the strength of the restraint.
        tightness : float
            b, tightness of the hyperbola around its minimum.
        qpotInp : list
            'q' vector
        listofelem : list
            list of elements whose charges are being fitted

        Returns
        -------
        np.ndarray
            Model 4 A matrix
        """
        newapot = copy.deepcopy(np.array(apotInp))
        N =  len(newapot)
        list_of_weights = []
        for i in range(N):
            if i in listofpolar_comb:
                list_of_weights.append(weight2)
            else:
                list_of_weights.append(weight1)

        if N != len(listofelem):
            print('List of elements should have the same size with A0 matrix.')
            return False
        for i in range(N):
            if listofelem[i] == 'H' or listofelem[i] == 1:
                continue
            else:
                newapot[i][i] += list_of_weights[i] / np.sqrt(qpotInp[i]**2 + tightness**2)
        return newapot, list_of_weights

    def Model4qpotFn(self, nAtoms, apot_comb, bpot_comb, weight1, weight2, tightness, elem_comb, atomid_comb,
                     atomidinfo, chargeinfo_comb, fixed_atomic_charge, equivGroupInp, listofpolar_comb):

        def Model4Iteration(qpot_temp):
            newapot, list_of_weights = self.Model4Amatrix(apot_comb, weight1, weight2, tightness, qpot_temp, elem_comb,  listofpolar_comb)

            apot_constrained, bpot_constrained = self.LagrangeChargeConstraint(newapot, bpot_comb, chargeinfo_comb)
            apot_sym, bpot_sym, elem_sym, atomid_sym, weights_sym= self.force_symmetry(apot_constrained, bpot_constrained, elem_comb, atomid_comb, list_of_weights, equivGroupInp)
            # consider fixed_atomic_charge
            qpot_sym = self.apply_fixed_atomic_charge(apot_sym, bpot_sym, elem_sym, atomid_sym, atomidinfo, fixed_atomic_charge, weights_sym, tightness)
            indices_sym = self.getCondensedIndices(len(apot_comb), equivGroupInp)
            qpot_expanded = self.Expandqpot(qpot_sym, indices_sym)
            return qpot_expanded, list_of_weights, apot_sym, bpot_sym, elem_sym, qpot_sym

        Size = len(apot_comb)
        qpotInitial = np.zeros((Size))
        for i in range(50):
            qpotNext, list_of_weights, apot_sym, bpot_sym, elem_sym, qpot_sym = Model4Iteration(qpotInitial)
            if np.linalg.norm(qpotNext-qpotInitial) < 1e-8: break
            qpotInitial = qpotNext.copy()
        #calculate condition number after fitting
        Options = {'weights':list_of_weights, 'tightness': tightness}
        cond1 = self.Ncond( apot_comb, elem_comb, qpotNext, Options)
        cond2 = self.Ncond( apot_sym, elem_sym, qpot_sym, Options)
        # print('###############################################################')
        # print('  After expansion:', cond1)
        # print('  Before expansion:', cond2)
        # print('###############################################################')
        # split qpot_Expanded into qpots for each molecule
        qpots = self.cut_qpot_comb(qpotNext, nAtoms)
        return qpots

    def Model4qpot(self, weight1, weight2, tightness):
        apots, bpots, atomid_comb, elem_comb, apot_comb, bpot_comb = self.combineMatrices()
        nAtoms = self.get_nAtoms(self.molecule.elems)
        equivGroup = self.get_equivGroup(atomid_comb)

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

        newlistofchargeinfo = self.combine_chargeinfo(self.molecule.listofchargeinfo, nAtoms)
        qpots = self.Model4qpotFn(nAtoms, apot_comb, bpot_comb, weight1, weight2, tightness,
                                  elem_comb, atomid_comb, self.molecule.atomidinfo,
                                  newlistofchargeinfo, self.inp.fixed_atomic_charge, equivGroup, listofpolar_comb)


        # write mol2 files with fitted charges.
        writeMol2 = False
        path = os.getcwd()
        if os.path.isdir('%s/resp_output' % path):
            print('\033[1;31m resp_output dir already exists!!! will overwrite anyway:/\x1b[0m')            
            writeMol2 = True
        else:
            writeMol2 = True
            os.mkdir('%s/resp_output' % path)
        # write text file for collecting result:)
        with open('%s/resp_output/result.txt' % path,'w') as f:
            f.write('respyte result\n')
            f.write(' Model 4')
            if self.inp.restraintinfo['matrices'] == ['esp', 'ef']:
                f.write(' RESPF\n')
            elif self.inp.restraintinfo['matrices'] == ['esp']:
                f.write(' RESP\n')
            elif self.inp.restraintinfo['matrices'] == ['ef']:
                f.write(' REF\n')
            f.write(' weight1 = %8.4f, weight2 = %8.4f, tightness = %8.4f\n' % (weight1, weight2, tightness))
            for idx, qpot in enumerate(qpots):
                f.write('mol%d\n' % (idx+1))
                for charge in qpot:
                    f.write('%8.4f\n' % round(charge,4))
        print()
        print('-------------------------------------------------------')
        print()
        print('              Model 4 with a1 = %.4f, a2 = %.4f' % (weight1, weight2))
        loc = 0
        esprrmss = []
        efrrmss  = []
        add = 0
        for idx, i in enumerate(self.molecule.nmols):
            if i == 0:
                add += 1
                esprrmss.append(0)
                efrrmss.append(0)
            else:
                config = 1
                for xyz,gridxyz, espval, efval  in zip(self.molecule.xyzs[loc: loc+i], self.molecule.gridxyzs[loc: loc+i], self.molecule.espvals[loc: loc+i], self.molecule.efvals[loc: loc+i]):
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
                    print(' MM dipole from model4: ', self.MMdipole(xyz, self.molecule.elems[idx], qpots[idx]))
                    print()
                    esprrms = self.espRRMS(apot, bpot, qpots[idx], espval)
                    efrrms  = self.efRRMS(Aef, Bef, qpots[idx], efval)
                    esprrmss.append(esprrms)
                    efrrmss.append(efrrms)
                    print(' espRRMS : ', "%.4f" % esprrms)
                    print(' efRRMS  : ', "%.4f\n" % efrrms)
                    if writeMol2 is True:
                        if self.inp.cheminformatics == 'openeye':
                            molt = 'oemol'
                        elif self.inp.cheminformatics == 'rdkit':
                            molt = 'rdmol'
                        else:
                            molt = 'fbmol'
                        self.write_output(self.molecule.mols[loc+config-1+ add], qpots[idx], moltype = molt,                                                                                            outfile = '%s/resp_output/mol%d_conf%d.mol2'% (path, idx+1, config))
                    config += 1
            loc += i
        return qpots, esprrmss, efrrmss

    def Model6Amatrix(self, apotInp, weight, tightness, qpotInp, listofelem,  listofburied):
        """
        Builds Model 4 A matrix from A0. No restraint on hydrogen atoms.

        Parameters
        ----------

        apotInp : np.ndarray
            "A" matrix; 2D array with dimension (# of atoms)
        weight1 : float
            a1, scaled factor(for buried atoms) which defines the asymptotic limits of the strength of the restraint.
        weight2 : float
            a2, scaled factor(for exposed atoms) which defines the asymptotic limits of the strength of the restraint.
        tightness : float
            b, tightness of the hyperbola around its minimum.
        qpotInp : list
            'q' vector
        listofelem : list
            list of elements whose charges are being fitted

        Returns
        -------
        np.ndarray
            Model 6 A matrix
        """
        newapot = copy.deepcopy(np.array(apotInp))
        N =  len(newapot)
        if N != len(listofelem):
            print('List of elements should have the same size with A0 matrix.')
            return False
        
        list_of_weights = []
        for i in range(N):
            if i in listofburied:
                list_of_weights.append(weight)
            else:
                list_of_weights.append(weight/10)
        for i in range(N):
            if listofelem[i] == 'H' or listofelem[i] == 1:
                continue
            else:
                newapot[i][i] += list_of_weights[i]/ np.sqrt(qpotInp[i]**2 + tightness**2)
        return newapot, list_of_weights

    def Model6qpotFn(self, nAtoms, apot_comb, bpot_comb, weight, tightness, elem_comb, atomid_comb,
                     atomidinfo, chargeinfo_comb, fixed_atomic_charge, equivGroupInp,  listofburied_comb):

        def Model6Iteration(qpot_temp):
            newapot, list_of_weights = self.Model6Amatrix(apot_comb, weight, tightness, qpot_temp, elem_comb,   listofburied_comb)
            apot_constrained, bpot_constrained = self.LagrangeChargeConstraint(newapot, bpot_comb, chargeinfo_comb)
            # Force symmetry based on the atomid
            apot_sym, bpot_sym, elem_sym, atomid_sym, weights_sym = self.force_symmetry(apot_constrained, bpot_constrained, elem_comb, atomid_comb, list_of_weights, equivGroupInp)
            # consider fixed_atomic_charge
            qpot_sym= self.apply_fixed_atomic_charge(apot_sym, bpot_sym, elem_sym, atomid_sym, atomidinfo, fixed_atomic_charge, weights_sym, tightness)
            indices_sym = self.getCondensedIndices(len(apot_comb), equivGroupInp)
            qpot_expanded = self.Expandqpot(qpot_sym, indices_sym)
            return qpot_expanded, list_of_weights, apot_sym, bpot_sym, elem_sym, qpot_sym

        Size = len(apot_comb)
        qpotInitial = np.zeros((Size))
        for i in range(50):
            qpotNext, list_of_weights, apot_sym, bpot_sym, elem_sym, qpot_sym = Model6Iteration(qpotInitial)
            if np.linalg.norm(qpotNext-qpotInitial) < 1e-8: break
            qpotInitial = qpotNext.copy()

        #calculate condition number after fitting
        Options = {'weights':list_of_weights, 'tightness': tightness}
        cond1 = self.Ncond( apot_comb, elem_comb, qpotNext, Options)
        cond2 = self.Ncond( apot_sym, elem_sym, qpot_sym, Options)
        # print('###############################################################')
        # print('  After expansion:', cond1)
        # print('  Before expansion:', cond2)
        # print('###############################################################')
        # split qpot_Expanded into qpots for each molecule
        qpots = self.cut_qpot_comb(qpotNext, nAtoms)
        return qpots

    def Model6qpot(self, weight, tightness):
        apots, bpots, atomid_comb, elem_comb, apot_comb, bpot_comb = self.combineMatrices()
        nAtoms = self.get_nAtoms(self.molecule.elems)
        equivGroup = self.get_equivGroup(atomid_comb)

        listofburied_comb = []
        loc = 0
        for idx, elems in enumerate(self.molecule.elems):
            if len(elems) == 0:
                continue
            else:
                for atmidx in self.molecule.listofburieds[idx]:
                    listofburied_comb.append(atmidx+loc)
                loc += len(elems)

        newlistofchargeinfo = self.combine_chargeinfo(self.molecule.listofchargeinfo, nAtoms)
        qpots = self.Model6qpotFn(nAtoms, apot_comb, bpot_comb, weight, tightness,
                                  elem_comb, atomid_comb, self.molecule.atomidinfo,
                                  newlistofchargeinfo, self.inp.fixed_atomic_charge, equivGroup, listofburied_comb)


        # write mol2 files with fitted charges.
        writeMol2 = False
        path = os.getcwd()
        if os.path.isdir('%s/resp_output' % path):
            print('\033[1;31m resp_output dir already exists!!! will overwrite anyway:/\x1b[0m')            
            writeMol2 = True
        else:
            writeMol2 = True
            os.mkdir('%s/resp_output' % path)
        # write text file for collecting result:)
        with open('%s/resp_output/result.txt' % path,'w') as f:
            f.write('respyte result\n')
            f.write(' Model 6')
            if self.inp.restraintinfo['matrices'] == ['esp', 'ef']:
                f.write(' RESPF\n')
            elif self.inp.restraintinfo['matrices'] == ['esp']:
                f.write(' RESP\n')
            elif self.inp.restraintinfo['matrices'] == ['ef']:
                f.write(' REF\n')
            f.write(' weight = %8.4f, tightness = %8.4f\n' % (weight, tightness))
            for idx, qpot in enumerate(qpots):
                f.write('mol%d\n' % (idx+1))
                for charge in qpot:
                    f.write('%8.4f\n' % round(charge,4))
        print()
        print('-------------------------------------------------------')
        print()
        print('              Model 6 with a = %.4f' % (weight))
        loc = 0
        esprrmss = []
        efrrmss  = []
        add = 0
        for idx, i in enumerate(self.molecule.nmols):
            if i == 0:
                add += 1
                esprrmss.append(0)
                efrrmss.append(0)
            else:
                config = 1
                for xyz,gridxyz, espval, efval  in zip(self.molecule.xyzs[loc: loc+i], self.molecule.gridxyzs[loc: loc+i], self.molecule.espvals[loc: loc+i], self.molecule.efvals[loc: loc+i]):
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
                    print(' MM dipole from model6: ', self.MMdipole(xyz, self.molecule.elems[idx], qpots[idx]))
                    print()
                    esprrms = self.espRRMS(apot, bpot, qpots[idx], espval)
                    efrrms  = self.efRRMS(Aef, Bef, qpots[idx], efval)
                    esprrmss.append(esprrms)
                    efrrmss.append(efrrms)
                    print(' espRRMS : ', "%.4f" % esprrms)
                    print(' efRRMS  : ', "%.4f\n" % efrrms)
                    if writeMol2 is True:
                        if self.inp.cheminformatics == 'openeye':
                            molt = 'oemol'
                        elif self.inp.cheminformatics == 'rdkit':
                            molt = 'rdmol'
                        else:
                            molt = 'fbmol'
                        self.write_output(self.molecule.mols[loc+config-1+ add], qpots[idx], moltype = molt,                                                                                            outfile = '%s/resp_output/mol%d_conf%d.mol2'% (path, idx+1, config))
                    config += 1
            loc += i
        return qpots, esprrmss, efrrmss

    def EspResiduals(self, qpotsInp):
        list_of_residuals = []
        if len(molecule.xyzs) == len(qpotsInp):
            qpots = qpotsInp
        else:
            qpots = []
            for idxx, Nconf in enumerate(molecule.nmols):
                for k in range(Nconf):
                    qpots.append(qpotsInp[idxx])
            if len(molecule.xyzs) != len(qpots):
                raise RuntimeError('sth wrong!!')
        for idx, xyzs in enumerate(self.molecule.xyzs):
            molbohr = xyzs
            gridxyzs = self.molecule.gridxyzs[idx]
            espval = self.molecule.espvals[idx]

            x = np.array([xyz[0] for xyz in molbohr])
            y = np.array([xyz[1] for xyz in molbohr])
            z = np.array([xyz[2] for xyz in molbohr])
            potv = np.array(espval)
            potvsq = potv*potv
            ssvpot = np.sum(potvsq)
            sspotsorted = np.sum(np.sort(potvsq))
            invRij = []
            dxij = []
            dyij = []
            dzij = []
            for pt in range(len(gridxyzs)):
                i = pt
                dxj = x - gridxyzs[i,0]
                dxij.append(dxj)
                dyj = y - gridxyzs[i,1]
                dyij.append(dyj)
                dzj = z - gridxyzs[i,2]
                dzij.append(dzj)
                rijsq = dxj*dxj + dyj*dyj + dzj*dzj
                invRjSq = 1.0/rijsq
                invRij.append(np.sqrt(invRjSq))

            vi = np.zeros(len(gridxyzs))
            for j, invRj in enumerate(invRij):
                vj = qpots[idx]*invRj
                vi[j] = np.sum(np.sort(vj))
            residual = vi - espval
            list_of_residuals.append(residual)
        return list_of_residuals



    def run(self):
        print(f'\n\033[1m Run respyte optimizer:\033[0m')
        if self.inp.restraintinfo:
            print(f'  * penalty function: \033[1m{self.inp.restraintinfo["penalty"]}\033[0m')
            if self.inp.restraintinfo['penalty'] == 'model2':
                aval = self.inp.restraintinfo['a']
                self.Model2qpot(aval)
            elif self.inp.restraintinfo['penalty'] == 'model3':
                aval = self.inp.restraintinfo['a']
                bval = self.inp.restraintinfo['b']
                self.Model3qpot(aval, bval)
            elif self.inp.restraintinfo['penalty'] == 'model4':
                a1val = self.inp.restraintinfo['a1']
                a2val = self.inp.restraintinfo['a2']
                bval = self.inp.restraintinfo['b']
                self.Model4qpot(a1val, a2val, bval)
            elif self.inp.restraintinfo['penalty'] == 'model5':
                a1val = self.inp.restraintinfo['a1']
                a2val = self.inp.restraintinfo['a2']
                bval = self.inp.restraintinfo['b']
                self.Model5qpot(a1val, a2val, bval)
            elif self.inp.restraintinfo['penalty'] == 'model6':
                aval = v.restraintinfo['a']
                bval = self.inp.restraintinfo['b']
                self.Model6qpot(aval, bval)
            elif self.inp.restraintinfo['penalty'] == '2-stg-fit':
                a1val = self.inp.restraintinfo['a1']
                a2val = self.inp.restraintinfo['a2']
                bval = self.inp.restraintinfo['b']
                self.twoStageFit(a1val, a2val, bval)
            elif self.inp.restraintinfo['penalty'] == '2-stg-fit(6)':
                aval = self.inp.restraintinfo['a']
                bval = self.inp.restraintinfo['b']
                self.twoStageFit_Model6(aval, bval)     
        else:
            print(f'  * penalty function:\033[1m 2-stg-fit\033[0m (default)')
            a1val = 0.0005
            a2val = 0.001
            bval = 0.1
            self.twoStageFit(a1val, a2val, bval)               