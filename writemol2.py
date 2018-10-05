import os,sys, copy
import math
import pandas as pd
import scipy.stats as stats
import scipy as sci
import numpy as np
import pylab
import re
from collections import OrderedDict, namedtuple, Counter
from readinp import Input
from warnings import warn
try:
    from forcebalance.Mol2 import *
except ImportError:
    warn('The Mol2 module cannot be imported (Cannot read/write Mol2 files)')

try:
    from forcebalance.PDB import *
except ImportError:
    warn('The pdb module cannot be imported (Cannot read/write PDB files)')
# Import CIB's resp_points python scripts to generate grid points:)

"""
For now, this code is mostly for protein. Will consider carbohydrates and organic compunds.

"""

class PdbtoMol2(object):
    def __init__(self, fnm = None, ftype = None):
        self.Read_Tab = {'pdb'    : self.read_pdb} # need to add xyz?? yes for sure... (not now)
        self.Write_Tab = {'mol2'  : self.write_mol2}
        # Data container.
        self.Data = {}
        if fnm is not None:
            self.Data['fnm'] = fnm
            if ftype is None:
                ftype = os.path.splitext(fnm)[1][1:]
            self.Data['ftype'] = ftype
            # Actually read the file:)
            Parsed = self.Read_Tab[ftype.lower()](fnm)
            for key, val in Parsed.items():
                self.Data[key] = val
        self.Write_Tab['mol2']()
    def read_pdb(self, fnm, **kwargs):
        """
        copied from molecule.py in forcebalance
        read_pdb : Loads a PDB and returns a dictionary containing its data.
        """
        F1 = open(fnm, 'r')
        ParsedPDB = readPDB(F1)
        Box = None
        # Separate into distinct lists for each model.
        PDBLines = [[]]
        # LPW: Keep a record of atoms which are followed by a terminal group.
        PDBTerms = []
        ReadTerms = True
        for x in ParsedPDB[0]:
            if x.__class__ in [END, ENDMDL]:
                PDBLines.append([])
                ReadTerms = False
            if x.__class__ in [ATOM, HETATM]:
                PDBLines[-1].append(x)
                if ReadTerms:
                    PDBTerms.append(0)
            if x.__class__ in [TER] and ReadTerms:
                PDBTerms[-1] = 1
            # if x.__class__ == CRYST1:
            #     Box = BuildLatticeFromLengthsAngles(x.a, x.b, x.c, x.alpha, x.beta, x.gamma)
        X=PDBLines[0]

        XYZ=np.array([[x.x,x.y,x.z] for x in X])/10.0#Convert to nanometers
        AltLoc=np.array([x.altLoc for x in X],'str') # Alternate location
        ICode=np.array([x.iCode for x in X],'str') # Insertion code
        ChainID=np.array([x.chainID for x in X],'str')
        AtomNames=np.array([x.name for x in X],'str')
        ResidueNames=np.array([x.resName for x in X],'str')
        ResidueID=np.array([x.resSeq for x in X],'int')
        # # LPW: Try not to number Residue IDs starting from 1...
        # if self.positive_resid: # If provided, enforce all positive resIDs.
        #     ResidueID=ResidueID-ResidueID[0]+1

        XYZList=[]
        for Model in PDBLines:
            # Skip over subsequent models with the wrong number of atoms.
            NewXYZ = []
            for x in Model:
                NewXYZ.append([x.x,x.y,x.z])
            if len(XYZList) == 0:
                XYZList.append(NewXYZ)
            elif len(XYZList) >= 1 and (np.array(NewXYZ).shape == np.array(XYZList[-1]).shape):
                XYZList.append(NewXYZ)

        if len(XYZList[-1])==0:#If PDB contains trailing END / ENDMDL, remove empty list
            XYZList.pop()

        # Build a list of chemical elements
        elem = []
        for i in range(len(AtomNames)):
            # QYD: try to use original element list
            if X[i].element:
                elem.append(X[i].element)
            else:
                thiselem = AtomNames[i]
                if len(thiselem) > 1:
                    thiselem = re.sub('^[0-9]','',thiselem)
                    thiselem = thiselem[0] + re.sub('[A-Z0-9]','',thiselem[1:])
                elem.append(thiselem)

        XYZList=list(np.array(XYZList).reshape((-1,len(ChainID),3)))

        bonds = []
        # Read in CONECT records.
        F2=open(fnm,'r')
        # QYD: Rewrite to support atom indices with 5 digits
        # i.e. CONECT143321433314334 -> 14332 connected to 14333 and 14334
        for line in F2:
            if line[:6] == "CONECT":
                conect_A = int(line[6:11]) - 1
                conect_B_list = []
                line_rest = line[11:]
                while line_rest.strip():
                    # Take 5 characters a time until run out of characters
                    conect_B_list.append(int(line_rest[:5]) - 1)
                    line_rest = line_rest[5:]
                for conect_B in conect_B_list:
                    bonds.append([conect_A, conect_B])

        Answer={"xyzs":XYZList, "chain":list(ChainID), "altloc":list(AltLoc), "icode":list(ICode),
                "atomname":[str(i) for i in AtomNames], "resid":list(ResidueID), "resname":list(ResidueNames),
                "elem":elem, "comms":['' for i in range(len(XYZList))], "terminal" : PDBTerms}
        # if len(bonds) > 0:
        #     self.top_settings["read_bonds"] = True
        #     Answer["bonds"] = bonds
        # if Box is not None:
        #     Answer["boxes"] = [Box for i in range(len(XYZList))]
        return Answer

    def write_mol2(self): # I doubt its robustness... Will be back later with fresh brain:/
        name = os.path.splitext(self.Data['fnm'])[0]
        # First check if 'name.mol2' exists in the working directory.
        row_format = "{:>10}" * 9
        if not os.path.exists('%s.mol2' % name):
            atomname = self.Data['atomname']
            resid = self.Data['resid']
            resname = self.Data['resname']
            elem = self.Data['elem']
            with open('%s.mol2' % name, 'w') as f:
                f.write('@<TRIPOS>MOLECULE\n')
                f.write('%s' % self.Data['comms'][0]+'\n')
                f.write('%4i' % len(self.Data['elem'])+'\n')
                f.write('PROTEIN\n') # need to change. hard-coded
                f.write('NO_CHARGE\n\n')
                f.write('@<TRIPOS>ATOM\n')
                for idx, i in enumerate(self.Data['xyzs'][0]):
                    atomidx = '%s%i' % (elem[idx], (idx+1))
                    f.write(row_format.format(idx+1, atomidx, i[0], i[1], i[2], atomname[idx], resid[idx], resname[idx], 0.0))
                    f.write('\n')
        else:
            print('Mol2 file alread exists?? Please check and rerun the code:)')

def main():
    # mol = PdbtoMol2('mol1.pdb')

    inp = Input()
    inp.readinput('input/respyte.yml')

    # Add coordinates
    for idx, i in enumerate(inp.nmols): # need to trim a bit more:
        molN = 'mol%d' % (idx+1)
        wkd = 'input/molecules/%s/' %(molN)
        coordfilepath = []
        #espffilepath = []
        if i > 1 and os.path.isfile(wkd + '%s.xyz' % (molN)): # In this case, xyz file contains mults conf.
            coordpath = wkd + '%s.xyz' % (molN)
            ftype = 'xyz'
            coordfilepath.append(coordpath)
            print('converting xyz to mol2 hasnt been implemented:/ Soooorryy.')
            raise NotImplementedError
        else:
            for j in range(i):
                confN = 'conf%d' % (j+1)
                path = wkd + '%s/' % (confN)
                for fnm in os.listdir(path):
                    if fnm.endswith('.xyz'):
                        coordpath = path + '%s_%s.xyz' % (molN, confN)
                        ftype = 'xyz'
                        print('converting xyz to mol2 hasnt been implemented:/ Soooorryy.')
                        raise NotImplementedError
                    elif fnm.endswith('.pdb'):
                        coordpath = path + '%s_%s.pdb' % (molN, confN)
                        ftype = 'pdb'
                        PdbtoMol2(coordpath)
                        print('Created %s_%s.mol2 in %s' % (molN, confN,path)) # it should be modified...

if __name__ == '__main__':
    main()
