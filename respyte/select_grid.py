from respyte.fbmolecule import * 
from respyte.fbmolecule import Molecule as FBMolecule

def SelectGridPts(pts, settings):

    # settings have fbmol, inner, outer, radiiType  
    if 'mol'  in settings: 
        mol = settings['mol']
        assert isinstance(mol, FBMolecule)
    else: 
        raise KeyError('fbmol should  be given for selecting pts' )
    
    if 'radiiType' in settings:
        if settings['radiiType'] == 'bondi':
            radiiType = BondiRadii
        elif settings['radiiType'] == 'Alvarez':
            radiiType = AlvarezRadii
    else: 
        radiiType = BondiRadii
    
    if  'inner' in settings:
        inner = settings['inner']
    else: 
        inner = 1.4

    if  'outer' in settings:
        outer = settings['outer']
    else: 
        outer = 2.0

    selectedPtsIdx = []
    selectedPts = []
    xyzs = mol.xyzs[0]
    innersSq = []
    outersSq = []
    atomicNum = [list(PeriodicTable.keys()).index(mol.elem[i])+1 for i in range(mol.na)]

    for elem in atomicNum:
        radii = float(radiiType[int(elem)-1])
        innersSq.append(radii*radii*(inner-1e-4)*(inner-1e-4))
        outersSq.append(radii*radii*(outer+1e-4)*(outer+1e-4))
    for idx, pt in enumerate(pts):
        goodPt = False
        for xyz, innerSq, outerSq in zip(xyzs, innersSq, outersSq):
            dx = pt[0]-xyz[0]
            dy = pt[1]-xyz[1]
            dz = pt[2]-xyz[2]
            distSq = dx*dx + dy*dy + dz*dz
            if distSq <= outerSq:
                goodPt = True
            if distSq < innerSq:
                goodPt = False
                break
        if goodPt:
            selectedPtsIdx.append(idx)
            selectedPts.append(pt)

    return selectedPtsIdx, selectedPts