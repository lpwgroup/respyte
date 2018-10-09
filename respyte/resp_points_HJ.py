#!/usr/bin/env python3
# (C) 2018 OpenEye Scientific Software Inc. All rights reserved.
#
# TERMS FOR USE OF SAMPLE CODE The software below ("Sample Code") is
# provided to current licensees or subscribers of OpenEye products or
# SaaS offerings (each a "Customer").
# Customer is hereby permitted to use, copy, and modify the Sample Code,
# subject to these terms. OpenEye claims no rights to Customer's
# modifications. Modification of Sample Code is at Customer's sole and
# exclusive risk. Sample Code may require Customer to have a then
# current license or subscription to the applicable OpenEye offering.
# THE SAMPLE CODE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED.  OPENEYE DISCLAIMS ALL WARRANTIES, INCLUDING, BUT
# NOT LIMITED TO, WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
# PARTICULAR PURPOSE AND NONINFRINGEMENT. In no event shall OpenEye be
# liable for any damages or liability in connection with the Sample Code
# or its use.

# Modified by HJ -2018-10-01

#############################################################################
# generates a shell of points around a molecule
#############################################################################
import sys, os
import numpy as np

import openeye.oechem as oechem
import gridgen.resp_points_utils as resp
from readinp import Input
from writemol2 import *

def ReadOEMolFromFile(filename):
    ifs = oechem.oemolistream()
    if not ifs.open(filename):
        oechem.OEThrow.Fatal("Cannot open input file %s!" % filename)
    mol = oechem.OEMol()
    if not oechem.OEReadMolecule(ifs, mol):
        oechem.OEThrow.Fatal("Unable to read molecule from %s" % filename)
    ifs.close()
    return mol

def PrintGridOptions( options):
    print('grid options:\n  grid type:', options['gridType'] )
    print('  inner boundary:', options['inner'] )
    print('  outer boundary:', options['outer'] )
    print('  grid spacing:', options['space'] )
    return

def main():
    inp = Input()
    inp.readinput('input/respyte.yml')

    # read grid_get setting from input
    gridType  = inp.gridinfo['type']

    if inp.gridinfo['radii']:
        radii = inp.gridinfo['radii']

    for idx, i in enumerate(inp.nmols): # need to trim a bit more:
        molN = 'mol%d' % (idx+1)
        wkd = 'input/molecules/%s/' %(molN)
        coordfilepath = []
        # for j in range(i):
        #     confN = 'conf%d' % (j+1)
        #     path = wkd + '%s/' % (confN)
# just added. if makes some issue, will change this part
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
                        print('Created %s_%s.mol2 in %s' % (molN, confN,path))
# Down to here:/ will see how it works...
                        molFile = path + '%s_%s.mol2' % (molN, confN)
                        mol = ReadOEMolFromFile(molFile)

                        if radii=='bondi':
                            oechem.OEAssignRadii(mol, oechem.OERadiiType_BondiVdw)
                        elif radii=='modbondi':
                            oechem.OEAssignRadii(mol, oechem.OERadiiType_BondiHVdw)
                        else:
                            oechem.OEThrow.Fatal('unrecognized radii type %s' % radii)

                    # Grid option gridType:
                    # * MSK: (Merz-Singh-Kollman) 4 shells: 1.4, 1.6, 1.8, 2.0 times vdW radius
                    # * shellConst: fcc grid where the boundaries are constants added to the vdw radius
                    # * shellFac: fcc grid where the boundaries are factors of the vdw radius (as with MSK)
                    # * shellFacConst: fcc grid where the inner boundaries is a factor of the vdw radius,
                    #   the outer boundaries is a constant (in Angstroms) added to the inner vdw radius
                    #   creating a vdw-based shell of constant thickness

                        # set grid options defaults to be overridden later if desired
                        gridOptions = {}
                        gridOptions['space'] = 0.7
                        gridOptions['inner'] = 1.4
                        gridOptions['outer'] = 2.0
                        if gridType=='MSK' or gridType=='msk':
                            gridOptions['gridType'] = 'MSK'
                            gridOptions['space'] = 1.0
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

                        if gridOptions['gridType']=='MSK':
                            # generate Merz-Singh-Kollman Connolly surfaces at 1.4, 1.6, 1.8, and 2.0 * vdW radii
                            allPts = resp.GenerateMSKShellPts( mol, gridOptions)
                        else:
                            # generate a face-centered cubic grid shell around the molecule using gridOptions
                            allPts = resp.GenerateGridPointSetAroundOEMol(mol, gridOptions)
                        print('Total points:', len(allPts))

                        ofs = open('%sgrid.dat' % path,'w')
                        for pt in allPts:
                            ofs.write( '{0:10.6f} {1:10.6f} {2:10.6f}\n'.format(pt[0], pt[1], pt[2]) )
                        ofs.close()

    return 0

if __name__ == "__main__":
    main()
