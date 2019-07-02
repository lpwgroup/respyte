"""
Unit and regression test for the respyte package.
"""

# Import package, test suite, and other packages as needed
import pytest
import sys, os
from distutils import dir_util
from pytest import fixture
from respyte.molecule import *
from respyte.molecule_resp import *

try:
    import openeye
    from openeye import oechem
except ImportError:
    pass
try:
    import rdkit.Chem as rdchem
except ImportError:
    pass

def test_respyte_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "respyte" in sys.modules

@pytest.mark.skipif("openeye" not in sys.modules, reason='openeye toolkit not found')
def test_openeye_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "openeye" in sys.modules

@pytest.mark.skipif("openeye" not in sys.modules, reason='openeye toolkit not found')
def test_Molecule_OEMol_addCoordFiles():
    orig_path = os.getcwd()
    this_file_folder = os.path.dirname(os.path.realpath(__file__))
    test_folder = os.path.join(this_file_folder, 'files')
    os.chdir(test_folder)

    def_mol = Molecule_OEMol()
    def_mol.addCoordFiles('meoh.pdb')

    assert def_mol.atomids == [[1, 2, 3, 4, 4, 4]]
    assert def_mol.listofpolars == [[0, 1]]
    assert def_mol.listofburieds == [[2, 3, 4, 5]]

    os.chdir(orig_path)

@pytest.mark.skipif("rdkit" not in sys.modules, reason='rdkit toolkit not found')
def test_rdkit_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "rdkit" in sys.modules

@pytest.mark.skipif("rdkit" not in sys.modules, reason='rdkit toolkit not found')
def test_Molecule_RDMol_addCoordFiles():
    orig_path = os.getcwd()
    this_file_folder = os.path.dirname(os.path.realpath(__file__))
    test_folder = os.path.join(this_file_folder, 'files')
    os.chdir(test_folder)

    def_mol = Molecule_RDMol()
    def_mol.addCoordFiles('meoh.pdb')

    assert def_mol.atomids == [[1, 2, 3, 4, 4, 4]]
    assert def_mol.listofpolars == [[0, 1]]
    assert def_mol.listofburieds == [[2, 3, 4, 5]]

    os.chdir(orig_path)
