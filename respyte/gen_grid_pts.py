#!/usr/bin/env python
#############################################################################
# Copyright (C) 2018 OpenEye Scientific Software, Inc.
#############################################################################
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

# Modified by HJ for the use in respyte package 2018-11-21

#############################################################################
# Grid Types
#  * shellConst: the cutoffs are constants added to the vdw radius
#  * shellFac: the cutoffs are factors of the vdw radius
#  * shellFacConst: the inner cutoff is a factor of the vdw radius,
#    the outer cutoff is a constant added to the inner vdw radius
#    creating a vdw-based shell of constant thickness
#  * MSK: (Merz-Singh-Kollman) 4 shells: 1.4, 1.6, 1.8, 2.0 times vdW radius
#
#############################################################################
#
# Grid Options object: a python dictionary of the following form:
#    gridOptions['gridType': string keyword for Grid Type, e.g. 'MSK'
#    gridOptions['space']: grid spacing (for grid) or inverse point density (for MSK)
#    gridOptions['inner']: float for inner vdw scale factor, e.g. 1.4 for MSK
#    gridOptions['outer']: float for outer vdw scale factor, e.g. 2.0 for MSK
#############################################################################


import os,sys
import math
import scipy as sci
import numpy as np
from warnings import warn
try:
    import rdkit.Chem as rdchem
except ImportError:
    warn(' The rdkit module cannot be imported. ' )
try:
    import openeye.oechem as oechem
except ImportError:
    warn(' The Openeye module cannot be imported. ')

# import molecule module copied from ForceBalance package
# just in case a user doesn't use any cheminformatics
from molecule import *

def GenerateVdwNeighborList( mol, atomi, scalefac, moltype = 'OEMol'):
    '''Scaling the atomic radii by scalefac, this function returns a list of OEAtoms
    in the OEMol mol which are neighbors (within sum of scaled radii) of the OEAtom atomi.
    Skip atoms with Atomic Number < 1 (dummy atoms)
    Input arguments:
      mol: the OEMol(or RDMol) whose atoms are examined to see if they are neighbors.
      atomi: the OEAtom in the OEMol whose neighbors we are looking for.
      scalefac: the scale factor for the atomic radii.
    Returns a list of OEAtoms belonging to the OEMol mol which are neighbors of atomi.'''
    if moltype is 'OEMol':
        xyzi = oechem.OEFloatArray(3)
        xyzj = oechem.OEFloatArray(3)
        scaledRadi = atomi.GetRadius()*scalefac
        # get xyz for atom i
        if not mol.GetCoords(atomi,xyzi):
            print('Molecule', mol.GetTitle(), 'cannot get coords for atom', atomi.GetName())
    elif moltype is 'RDMol':
        xyzi = np.zeros(3) # dont need
        xyzj = np.zeros(3) # dont need
        scaledRadi = float(atomi.GetProp('radius'))*scalefac
        # get xyz for atom i
        pos = mol.GetConformer().GetPositions()
        xyzi = pos[atomi.GetIdx()]
    #print('atom i  is', atomi.GetName(), xyzi)
    nbrs = []
    for atomj in mol.GetAtoms():
        if atomj.GetAtomicNum()<1:
            # dummy atoms are ignored
            continue
        # skip if this atom i and atom j are the same atom
        if atomi.GetIdx()==atomj.GetIdx():
            continue
        # get xyz for atom j
        if moltype is 'OEMol':
            if not mol.GetCoords(atomj,xyzj):
                print('Molecule', mol.GetTitle(), 'cannot get coords for atom', atomj.GetName())
        elif moltype is 'RDMol':
            xyzj = pos[atomj.GetIdx()]
        #print('atom j  is', atomj.GetName(), xyzj)
        # calc the squared distance
        dx = xyzi[0]-xyzj[0]
        dy = xyzi[1]-xyzj[1]
        dz = xyzi[2]-xyzj[2]
        distSq = dx*dx + dy*dy + dz*dz
        # cutoff distance is sum of vdW radii; within this the two atoms are neighbours
        if moltype is 'OEMol':
            scaledRadj = atomj.GetRadius()*scalefac
        elif moltype is 'RDMol':
            scaledRadj = float(atomj.GetProp('radius'))*scalefac
        cutoff = scaledRadi+scaledRadj
        #print(mol.GetTitle(),atomi.GetName(), atomj.GetName(), cutoff, np.sqrt(distSq) )
        # within cutoff the two atoms are neighbours
        if distSq<=(cutoff*cutoff):
            nbrs.append( atomj)
            #print('  atoms',atomi.GetName(), atomj.GetName(), 'distance',
            #      np.sqrt(distSq), 'are within vdW dist', cutoff)

    #print('for', atomi.GetName(), 'neighbors are:', [atom.GetName() for atom in nbrs])
    return nbrs

def CullClashPts( mol, nbrList, scalefac, pts, moltype = 'OEMol'):
    '''Culls out points in the list which are within scaled vdw radii of any other
    atom in the OEMol mol.
    Input arguments:
      mol: the OEMol(or RDMol) whose neighbors are examined.
      nbrList: a list of OEAtoms which are neighbors of the point list.
      scalefac: the scale factor for the atomic radii.
      pts: the list of points to cull
    Returns the list of points outside the scaled radii of the neighboring atoms.'''
    # precalculate neighbor coords
    nbrXyz = []
    if moltype is 'OEMol':
        xyztmp = oechem.OEFloatArray(3)
        for atom in nbrList:
            if not mol.GetCoords(atom,xyztmp):
                print('cannot get coords for atom', atom.GetName())
            nbrXyz.append( list(xyztmp) )
        # precalculate neighbor scaled squared radii
        nbrRadiusSq = [ atom.GetRadius()*scalefac*atom.GetRadius()*scalefac for atom in nbrList]
    elif moltype is 'RDMol':
        xyztmp = np.zeros(3)
        pos = mol.GetConformer().GetPositions()
        for atom in nbrList:
            xyztmp = pos[atom.GetIdx()]
            nbrXyz.append( list(xyztmp) )
        # precalculate neighbor scaled squared radii
        nbrRadiusSq = [ float(atom.GetProp('radius'))*scalefac*float(atom.GetProp('radius'))*scalefac for atom in nbrList]

    culled = []
    # iterate over each point in the point list
    for pt in pts:
        #print(pt)
        goodPt = True
        # iterate over all atoms in the neighborlist; precalculated coords and radiusSq
        for i, (xyz, radiusSq) in enumerate( zip(nbrXyz, nbrRadiusSq)):
            #print(xyz)
            dx = pt[0]-xyz[0]
            dy = pt[1]-xyz[1]
            dz = pt[2]-xyz[2]
            distSq = dx*dx + dy*dy + dz*dz
            #print( i, distSq, radiusSq)
            if distSq<radiusSq:
                #print('bad point; breaking')
                goodPt = False
                break
        if goodPt:
            #print('good point; adding')
            culled.append(pt)
    return culled


def GenerateConnollySphere( radius, xyz, density=1.0):
    '''Generates a Connolly sphere of points of specified radius around
    center xyz and having approximately the specified surface area per point.
    Input arguments:
      radius: the radius of the sphere.
      xyz: the coordinates of the center of the sphere.
      density: the targeted surface area per point.
    Returns the list of points forming the sphere.'''
    #print(atom.GetName(), 'radius and xyz:', radius, xyz)
    pi = np.pi
    npts = int(4*pi*radius*radius*density)
    nequat = int(np.sqrt(npts*pi))
    nvert = int(nequat/2)
    #print( 'npts, nequat, nvert:', npts, nequat, nvert)
    pts = []
    for i in range(nvert+1):
        fi = pi*i/nvert
        z = np.cos(fi)*radius
        xy = np.sin(fi)
        nhor = max( int(nequat*xy), 1)
        #print('  i, nhor, fi, z, xy:', i, nhor, fi, z, xy)
        for j in range(1,nhor+1):
            fj = 2*pi*j/nhor
            x = np.cos(fj)*xy*radius
            y = np.sin(fj)*xy*radius
            pts.append( (x+xyz[0], y+xyz[1], z+xyz[2]) )
            #print('    j, xyz:', j, (x+xyz[0], y+xyz[1], z+xyz[2]) )
    return pts

def GenerateMSKShellPts( mol, options, moltype='OEMol'):
    '''Generates a set of points around the molecule, based on scaled atomic radii,
    according to the Merz-Singh-Kollman (MSK) scheme for ESP points. The MSK scheme
    is based on Bondi atomic radii, with four Connolly-sphere shells of points set at
    scaled atomic radii with scale factors 1.4, 1.6, 1.8, and 2.0 with a targeted
    surface density of 1 point per angstrom squared on each shell. This function offers
    flexibility in that it uses the atomic radii on the molecule and the options object
    to specify target surface spacing (inverse of density) and lower and upper scale
    factors (incrementing by 0.2 for each shell).
    Input arguments:
      mol: the OEMol(or RDMol) molecule for which we want the MSK point set.
      options: a dictionary with keys "space", "inner", and "outer", pointing to
        values for the target surface spacing, inner and outer vdw scale factors,
        respectively, all in angstrom units.
    Returns the MSK set as a list of xyz triples (in angstroms)'''
    density = 1/options['space']
    print('MSK points with density', density)
    if moltype is 'OEMol':
        xyz = oechem.OEFloatArray(3)
    elif moltype is 'RDMol':
        xyz = np.zeros(3)
        pos = mol.GetConformer().GetPositions()
    molPts = []
    shellScale = options['inner']
    # increment atom radius shellScale by 0.2 placing a shell of points each time
    # the "while" test has a tiny increment to prevent premature termination
    while shellScale<(options['outer']+.000001):
        shellPts = []
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum()<1:
                # dummy atoms are ignored
                continue
            if moltype is 'OEMol':
                if not mol.GetCoords(atom,xyz):
                    print('cannot get coords for atom', atom.GetName())
                scaledRad = atom.GetRadius()*shellScale
            elif moltype is 'RDMol':
                xyz = pos[atom.GetIdx()]
                scaledRad =  float(atom.GetProp('radius'))*shellScale
            nbrs = GenerateVdwNeighborList(mol, atom, shellScale, moltype)
            atomPts = GenerateConnollySphere( scaledRad, xyz, density)
            culledPts = CullClashPts( mol, nbrs, shellScale, atomPts, moltype)
            #print( atom.GetName(), atom.GetRadius(), 'numPts pre-cull:', len(atomPts), 'post-cull:', len(culledPts))
            shellPts += culledPts
        print('MSK shell scaling vdW by {0:5.2f}: {1:d} points'.format(shellScale,len(shellPts)) )
        # molPts += shellPts
        molPts.append(shellPts) # changed
        shellScale += 0.2

    return molPts

def GenerateCutoffs( mol, options, moltype = 'OEMol'):
    '''Sets inner and outer boundary radii on each OEAtom as prescribed by the options.
    Input arguments:
      mol: the OEMol(or RDMol) molecule for which the atomic boundary radii will be set.
      options: a dictionary with keys "gridType", "inner", and "outer", pointing to
        values for the grid (ie what kind of boundaries) and inner and outer vdw
        scale factors, respectively.
    Returns the OEMol with the inner and outer boundary set on each atom as
      Generic Data with tags "PointSetInnerRad" and "PointSetInnerRad", respectively.
    '''
    vdwInner = []
    vdwOuter = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum()<1:
            # dummy atoms are ignored
            continue
        if moltype is 'OEMol':
            radius = atom.GetRadius()
        elif moltype is 'RDMol':
            radius = float(atom.GetProp('radius'))
            # print('I must be printed out!')
            # print(radius)
            # input()
        if options['gridType']=='MSK':
            # the inner cutoff is 1.4*vdwRadius (outer set to 2.0 but not used)
            inner = radius*1.4
            outer = radius*2.0
        if options['gridType']=='extendedMsk':
            # the inner cutoff is 1.4*vdwRadius (outer set to 2.0 but not used)
            inner = radius*0.8 # changed
            outer = radius*2.4

        elif options['gridType']=='shellFac':
            # the cutoffs are factors of the vdw radius
            inner = radius*options['inner']
            outer = radius*options['outer']
        elif options['gridType']=='shellConst':
            # the inner cutoff is a constant added to the vdw radius
            # the outer cutoff is a constant added to the inner cutoff
            # creating a vdw-based shell of constant thickness
            inner = radius+options['inner']
            outer = radius+options['outer']
        elif options['gridType']=='shellFacConst':
            # the inner cutoff is a factor of the vdw radius
            # the outer cutoff is a constant added to the inner cutoff
            # creating a vdw-based shell of constant thickness
            inner = radius*options['inner']
            outer = inner+options['outer']
        else:
            print('Not an allowed Grid Type option:', options['gridType'])
            return None
        if moltype is 'OEMol':
            atom.SetData('PointSetInnerRad', inner)
            atom.SetData('PointSetOuterRad', outer)
        elif moltype is 'RDMol':
            atom.SetProp('PointSetInnerRad', str(inner))
            atom.SetProp('PointSetOuterRad', str(outer))
    return mol

def GenerateBoxMinMax( mol, moltype = 'OEMol'):
    '''finds the smallest box dimension to enclose the molecule around the
    outer boundary radius of all the atoms. Each atom's outer boundary radius
    is stored in the atom's Generic Data with tag "PointSetOuterRad".
    Input arguments:
      mol: the OEMol(or RDMol) molecule for which the box dimensions are desired.
    Returns two tuples of xyz coords, the first is the box minimum coords and
      the second is the the box maximum coords.'''
    if moltype is 'OEMol':
        xyzOEArr = oechem.OEFloatArray(3)
    elif moltype is 'RDMol':
        xyzOEArr = np.zeros(3)
        pos = mol.GetConformer().GetPositions()
        # print(pos)
        # input()
    boxmin = np.array([1.e10, 1.e10, 1.e10])
    boxmax = np.array([-1.e10, -1.e10, -1.e10])
    for atom in mol.GetAtoms():
        # dummy atoms are ignored
        atomicNum = atom.GetAtomicNum()
        if atomicNum<1:
            continue
        # get xyz coords for atom
        if moltype is 'OEMol':
            if not mol.GetCoords(atom,xyzOEArr):
                print('Molecule', mol.GetTitle(), 'cannot get coords for atom', atom.GetName())
            xyz = np.array(xyzOEArr)
            # get outer (larger) radius for atom to use for outer limit of box
            outer = atom.GetRadius()
            if atom.HasData('PointSetOuterRad'):
                outer = atom.GetData('PointSetOuterRad')
        elif moltype is 'RDMol':
            xyzOEArr = pos[atom.GetIdx()]
            xyz = np.array(xyzOEArr)
            outer = float(atom.GetProp('radius'))
            if atom.GetProp('PointSetOuterRad') is not None:
                outer = float(atom.GetProp('PointSetOuterRad'))
        # find atom min and max vdw
        atomMin = xyz-outer
        atomMax = xyz+outer
        for crd in [0,1,2]:
            if atomMin[crd]<boxmin[crd]:
                boxmin[crd]=atomMin[crd]
            if atomMax[crd]>boxmax[crd]:
                boxmax[crd]=atomMax[crd]
    return boxmin, boxmax

def GenerateFaceCenteredGrid( boxmin, boxmax, space=0.5):
    '''Generates a Face-Centered Cubic (FCC) Grid within the cubic box delineated
    by xyz coords boxmin and boxmax, with spacing "space" between grid points.
    Note that this is the full FCC grid with no points removed around the molecule.
    Input arguments:
      boxmin, boxmax: two tuples of xyz coords, the first is the box minimum
        coords and the second is the the box maximum coords.
      space: the spacing between grid points.
    Returns a list of triples of floats, each one the xyz coords of a grid
      point in the Face-Centered Cubic Grid enclosing the molecule.
'''
    # factors pertaining to grid spacing
    root2= np.sqrt(2.0)
    dx = space*root2
    delta= space/root2
    #
    pts = []
    # loop over z
    ztoggle = False
    z = boxmin[2]
    while (z<boxmax[2]):
        ztoggle = not ztoggle
        if ztoggle:
            ytoggle = False
        else:
            ytoggle = True
        # loop over y
        y = boxmin[1]
        while (y<boxmax[1]):
            ytoggle = not ytoggle
            if ytoggle:
                x = boxmin[0]+delta
            else:
                x = boxmin[0]
            # loop over x
            while (x<boxmax[0]):
                #print(x,y,z)
                pts.append( (x,y,z) )
                x += dx
            y += delta
        z += delta

    return pts

def CullGridPts( mol, pts, moltype = 'OEMol'):
    '''Cull a grid of points around a molecule, keeping only points which are
    inside the outer radius of any atom and outside the inner radius of all atoms,
    thus forming a shell arund the molecule.
    The inner and outer boundary are found on each atom as Generic Data with
    tags "PointSetInnerRad" and "PointSetInnerRad", respectively.
    Input arguments:
      mol: the OEMol(or RDMol) with inner and outer boundaries set on each atom.
      pts: the list of points to cull
    Returns the list of points forming a shell around the molecule.'''
    # Preprocess to get simple iterables for the atom info
    if moltype is 'OEMol':
        xyzOEArr = oechem.OEFloatArray(3)
    elif moltype is 'RDMol':
        xyzOEArr = np.zeros(3)
        pos = mol.GetConformer().GetPositions()

    innersSq = []
    outersSq = []
    xyzs = []
    if moltype is 'OEMol':
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum()<1:
                # dummy atoms are ignored
                continue
            if not atom.HasData('PointSetInnerRad'):
                print('Molecule', mol.GetTitle(),'no PointSetInnerRad')
                return None
            innersSq.append( atom.GetData('PointSetInnerRad')*atom.GetData('PointSetInnerRad'))
            if not atom.HasData('PointSetOuterRad'):
                print('Molecule', mol.GetTitle(),'no PointSetOuterRad')
                return None
            outersSq.append( atom.GetData('PointSetOuterRad')*atom.GetData('PointSetOuterRad'))
            if not mol.GetCoords(atom,xyzOEArr):
                print('Molecule', mol.GetTitle(),'cannot get coords for atom', atom.GetName())
                return None
            xyzs.append( list(xyzOEArr))
    elif moltype is 'RDMol':
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum()<1:
                # dummy atoms are ignored
                continue
            innersSq.append(float(atom.GetProp('PointSetInnerRad'))*float(atom.GetProp('PointSetInnerRad')))
            outersSq.append(float(atom.GetProp('PointSetOuterRad'))*float(atom.GetProp('PointSetOuterRad')))
            xyzOEArr = pos[atom.GetIdx()]
            xyzs.append(list(xyzOEArr))
    culled = []
    for pt in pts:
        goodPt = False
        for xyz, innerSq, outerSq in zip(xyzs,innersSq,outersSq):
            dx = pt[0]-xyz[0]
            dy = pt[1]-xyz[1]
            dz = pt[2]-xyz[2]
            distSq = dx*dx + dy*dy + dz*dz
            if distSq<outerSq:
                #print('good point: inside an outer radius')
                goodPt = True
            if distSq<innerSq:
                #print('bad point; breaking')
                goodPt = False
                break
        if goodPt:
            #print('good point; adding')
            culled.append(pt)
    return culled

def GenerateGridPointSetAroundOEMol(mol, options, moltype='OEMol'):
    '''Generates a Face-Centered Cubic (FCC) Grid around a molecule.
    The grid is pruned to form a shell around the molecule, based on atomic radii
    found on the molecule. The pruning is affected by the options object, a python
    dictionary specifying the following keys:
      "inner": a float used to establish the inner boundary.
      "outer": a float used to establish the outer boundary.
      "space": the spacing between grid points.
      "gridType": one of the following keyword strings used to determine how
         inner and outer will be applied to amke inner and outer boundaries::
         "shellConst": inner and outer are constants added to the vdw radius.
         "shellFac": inner and outer are factors of the vdw radius.
         "shellFacConst": inner is a factors of the vdw radius and outer is a
            a constant added to the inner vdw boundary.
    Input arguments:
      mol: the OEMol molecule for which we want the FCC grid.
      options: a dictionary with keys as outlined above.
    Returns the FCC Grid shell as a list of xyz triples (in angstroms)'''
    if options['gridType']=='shellConst':
        # the cutoffs are constants added to the vdw radius
        innerstr = 'vdw radius +'
        outerstr = 'inner boundary +'
    elif options['gridType']=='shellFacConst':
        # the inner cutoff is a factor of the vdw radius
        # the outer cutoff is a constant added to the inner vdw radius
        # creating a vdw-based shell of constant thickness
        innerstr = 'vdw radius *'
        outerstr = 'inner boundary +'
    elif options['gridType']=='shellFac':
        # the cutoffs are factors of the vdw radius
        innerstr = 'vdw radius *'
        outerstr = 'vdw radius *'
    else:
        print('Not an allowed Grid Type option:', options['gridType'])
        return None
    print('Face-Centered Cubic grid shell with density', options['space'] )
    print('  Inner boundary is %s %.2f' % (innerstr, options['inner'] ))
    print('  Outer boundary is %s %.2f' % (outerstr, options['outer'] ))
    # generate cutoffs, setting them as data on the atoms
    molPrep = GenerateCutoffs( mol, options, moltype)
    # based on the cutoffs, get the min and max box dimensions for the grid
    boxmin, boxmax = GenerateBoxMinMax(molPrep, moltype)
    # Generate face centered cubic grid within the box
    gridPts = GenerateFaceCenteredGrid( boxmin,  boxmax, options['space'] )
    # now cull the grid to fit between inner and outer radii of the atoms
    cullPts = CullGridPts( molPrep, gridPts, moltype)
    return cullPts

def PrintGridOptions( options):
    print('grid options:\n  grid type:', options['gridType'] )
    print('  inner boundary:', options['inner'] )
    print('  outer boundary:', options['outer'] )
    print('  grid spacing:', options['space'] )
    return
