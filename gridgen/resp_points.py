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

#############################################################################
# generates a shell of points around a molecule
#############################################################################
import sys
import numpy as np

import openeye.oechem as oechem
import resp_points_utils as resp

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

def main(argv=[__name__]):

    itf = oechem.OEInterface()
    oechem.OEConfigure(itf, InterfaceData)
    if not oechem.OEParseCommandLine(itf, argv):
        oechem.OEThrow.Fatal("Unable to interpret command line!")

    molFile = itf.GetString("-in")
    outFilePrefix = itf.GetString("-prefix")

    gridType = itf.GetString("-type")

    mol = ReadOEMolFromFile(molFile)

    radii = itf.GetString("-radii")
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

    if itf.HasFloat("-inner"):
        gridOptions['inner'] = itf.GetFloat("-inner")
    if itf.HasFloat("-outer"):
        gridOptions['outer'] = itf.GetFloat("-outer")
    if itf.HasFloat("-space"):
        gridOptions['space'] = itf.GetFloat("-space")

    #Debugging: print options set in the options dictionary
    #PrintGridOptions( gridOptions)

    if gridOptions['gridType']=='MSK':
        # generate Merz-Singh-Kollman Connolly surfaces at 1.4, 1.6, 1.8, and 2.0 * vdW radii
        allPts = resp.GenerateMSKShellPts( mol, gridOptions)
    else:
        # generate a face-centered cubic grid shell around the molecule using gridOptions
        allPts = resp.GenerateGridPointSetAroundOEMol(mol, gridOptions)
    print('Total points:', len(allPts))

    ofs = open(outFilePrefix+'.pointsxyz','w')
    for pt in allPts:
        ofs.write( '{0:10.6f} {1:10.6f} {2:10.6f}\n'.format(pt[0], pt[1], pt[2]) )
    ofs.close()

    return 0

#############################################################################
# INTERFACE
#############################################################################


InterfaceData = '''
!BRIEF -in <molecule> -prefix <outFilePrefix> -type <gridType>

  !PARAMETER -in
    !ALIAS -i
    !TYPE string
    !REQUIRED true
    !VISIBILITY simple
    !BRIEF Input filename of the molecule
  !END

  !PARAMETER -prefix
    !ALIAS -p
    !TYPE string
    !REQUIRED true
    !BRIEF Prefix for output file names (prefix.pointsxyz)
  !END

  !PARAMETER -type
    !TYPE string
    !REQUIRED false
    !VISIBILITY simple
    !DEFAULT MSK
    !LEGAL_VALUE MSK
    !LEGAL_VALUE msk
    !LEGAL_VALUE fcc
    !LEGAL_VALUE vdwfactors
    !LEGAL_VALUE vdwconstants
    !BRIEF Type of shell points desired (e.g. MSK, fcc)
  !END

  !PARAMETER -radii
    !TYPE string
    !REQUIRED false
    !VISIBILITY simple
    !DEFAULT bondi
    !LEGAL_VALUE bondi
    !LEGAL_VALUE modbondi
    !BRIEF Type of atomic radii to use for inner and outer boundaries of the shell.
  !END

  !PARAMETER -inner
    !TYPE float
    !LEGAL_RANGE 0.0 10.0
    !REQUIRED false
    !VISIBILITY simple
    !BRIEF factor or constant to use with atomic radii for the inner boundary.
  !END

  !PARAMETER -outer
    !TYPE float
    !LEGAL_RANGE 0.0 10.0
    !REQUIRED false
    !VISIBILITY simple
    !BRIEF factor or constant to use with atomic radii for the outer boundary.
  !END

  !PARAMETER -space
    !TYPE float
    !LEGAL_RANGE 0.01 100.0
    !REQUIRED false
    !VISIBILITY simple
    !BRIEF spacing of points (for grid) or inverse point density (for MSK).
  !END

!END
'''

if __name__ == "__main__":
    sys.exit(main(sys.argv))
