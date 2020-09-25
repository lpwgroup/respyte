from respyte.molecule import *

BondiRadii = [1.2, 1.4, # exchanged None to 2.0
              1.81, 2.00, 2.00, 1.70, 1.55, 1.52, 1.47, 1.54,
              2.27, 1.73, 2.00, 2.22, 1.80, 1.80, 1.75, 1.88,
              2.75, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00, 1.63, 1.40, 1.39, 1.87, 2.00, 1.85, 1.90, 1.83, 2.02,
              2.00, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00, 1.63, 1.72, 1.62, 1.93, 2.17, 2.00, 2.00, 1.98, 2.16]
# from oechem
BondiRadiiOechem = [1.20, 1.40,
                    1.82, 2.00, 2.00, 1.70, 1.55, 1.52, 1.47, 1.54,
                    2.27, 1.73, 2.00, 2.01, 1.80, 1.80, 1.75, 1.88,
                    2.75, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00, 1.63, 1.40, 1.39, 1.87, 2.00, 1.85, 1.90, 1.85, 2.02,
                    2.00, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00, 1.72, 1.58, 2.00, 2.17, 2.00, 2.06, 1.98, 2.16]

AlvarezRadii = [1.20, 1.43,
                2.12, 1.98, 1.91, 1.77, 1.66, 1.50, 1.46, 1.58,
                2.50, 2.51, 2.25, 2.19, 1.90, 1.89, 1.82, 1.83,
                2.73, 2.62, 2.58, 2.46, 2.42, 2.45, 2.45, 2.44, 2.40, 2.40, 2.38, 2.39, 2.32, 2.29, 1.88, 1.82, 1.86, 2.25,
                3.21, 2.84, 2.75, 2.52, 2.56, 2.45, 2.44, 2.46, 2.44, 2.15, 2.53, 2.49, 2.43, 2.42, 2.47, 1.99, 2.04, 2.06]
PolarHradii = 0.95 # in Angstrom
bohr2Ang = 0.52918825


def SelectGridPts(mol, inner, outer, pts, radiiType):
    if radiiType == 'bondi':
        # radiiType = BondiRadii
        radiiType = BondiRadiiOechem
    elif radiiType == 'Alvarez':
        radiiType = AlvarezRadii
    culled = []
    selectedPts = []
    xyzs = mol.xyzs[0]
    innersSq = []
    outersSq = []
    atomicNum = [list(PeriodicTable.keys()).index(mol.elem[i])+1 for i in range(mol.na)]

    for elem in atomicNum:
        radii = float(radiiType[int(elem)-1])
        innersSq.append(radii*radii*inner*inner)
        outersSq.append(radii*radii*outer*outer)
    for idx, pt in enumerate(pts):
        goodPt = False
        for xyz, innerSq, outerSq in zip(xyzs, innersSq, outersSq):
            dx = pt[0]-xyz[0]
            dy = pt[1]-xyz[1]
            dz = pt[2]-xyz[2]
            distSq = dx*dx + dy*dy + dz*dz
            if distSq < outerSq:
                goodPt = True
            if distSq < innerSq:
                goodPt = False
                break
        if goodPt:
            culled.append(idx)
            selectedPts.append(pt)

    return culled, selectedPts

