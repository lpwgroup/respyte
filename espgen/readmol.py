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

BondiRadii = [1.2, 1.4, # exchanged None to 2.0
              1.81, 2.0, 2.0, 1.70, 1.55, 1.52, 1.47, 1.54,
              2.27, 1.73, 2.0, 2.22, 1.80, 1.80, 1.75, 1.88,
              2.75, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.63, 1.40, 1.39, 1.87, 2.0, 1.85, 1.90, 1.83, 2.02]
# from oechem
BondiRadiiOechem = [1.2, 1.4,
                    1.82, 2.0, 2.0, 1.70, 1.55, 1.52, 1.47, 1.54,
                    2.27, 1.73, 2.0, 2.1, 1.80, 1.80, 1.75, 1.88,
                    2.75, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.63, 1.40, 1.39, 1.87, 2.0, 1.85, 1.90, 1.85, 2.02]

AlvarezRadii = [1.20, 1.43,
                2.12, 1.98, 1.91, 1.77, 1.66, 1.50, 1.46, 1.58,
                2.50, 2.51, 2.25, 2.19, 1.90, 1.89, 1.82, 1.83,
                2.73, 2.62, 2.58, 2.46, 2.42, 2.45, 2.45, 2.44, 2.40, 2.40, 2.38, 2.39, 2.32, 2.29, 1.88, 1.82, 1.86, 2.25]

def ReadOEMolFromFile(filename):
    """
    Construct a oemol object from a input file (Copied from Christopher's code)
    """
    ifs = oechem.oemolistream()
    if not ifs.open(filename):
        oechem.OEThrow.Fatal("Cannot open input file %s!" % filename)
    mol = oechem.OEMol()
    if not oechem.OEReadMolecule(ifs, mol):
        oechem.OEThrow.Fatal("Unable to read molecule from %s" % filename)
    ifs.close()
    return mol

def ReadRdMolFromFile(filename):
    if filename.endswith('.mol2'):
        mol = rdchem.MolFromMol2File(filename, removeHs=False)
    elif filename.endswith('.pdb'):
        mol = rdchem.MolFromPDBFile(filename, removeHs=False)
    else:
        raise RuntimeError('The extension of input file should be either mol2 or pdb!')
    return mol

def ReadMolFromFile(filename):
    try:
        mol = Molecule(filename)
    except:
        raise RuntimeError("Unable to read molecule from %s" % filename)
    return mol

"""
Adding vdw radii set to molecule object
"""

def assignRadii(mol, molType='FBMol', radii='bondi'):
    if molType is 'OEMol':
        if radii=='bondi':
            oechem.OEAssignRadii(mol, oechem.OERadiiType_BondiVdw) # atom.GetRadius()
        elif radii=='modbondi':
            oechem.OEAssignRadii(mol, oechem.OERadiiType_BondiHVdw)
        elif radii == 'Alvarez':
            for atom in mol.GetAtoms():
                atom.SetRadius(float(AlvarezRadii[atom.GetAtomicNum()-1]))
        else:
            oechem.OEThrow.Fatal('unrecognized radii type %s' % radii)

    elif molType is 'RDMol':
        if radii =='bondi':
            for atom in mol.GetAtoms():
                atom.SetProp('radius',str(BondiRadii[atom.GetAtomicNum()-1])) # atom.GetProp('radius')
        elif radii =='Alvarez':
            for atom in mol.GetAtoms():
                atom.SetProp('radius',str(AlvarezRadii[atom.GetAtomicNum()-1]))
        elif radii == 'modbondi':
            raise NotImplementedError('assigning modified bondi radii on RDKit mol not implemented yet!')

    elif molType is 'FBMol':
        if radii == 'bondi':
            mol.add_vdw_radii('bondi') # mol.vdwradii
        elif radii == 'Alvarez' :
            mol.add_vdw_radii('Alvarez')
        else:
            raise NotImplementedError('not implemented yet!')

    return mol
