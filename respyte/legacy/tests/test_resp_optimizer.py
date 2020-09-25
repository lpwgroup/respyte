"""
Unit and regression test for the respyte package.
"""

# Import package, test suite, and other packages as needed
import pytest
import sys, os, shutil
from distutils import dir_util
from pytest import fixture
from respyte.molecule import *
from respyte.molecule_resp import *
from respyte.resp_unit import *

# test if the package successfully reproduce 1993 org paper
@pytest.mark.skipif("openeye" not in sys.modules, reason='openeye toolkit not found')
def test_optimizer():
    orig_path = os.getcwd()
    this_file_folder = os.path.dirname(os.path.realpath(__file__))
    test_folder = os.path.join(this_file_folder, 'files')
    os.chdir(test_folder)

    inp = Input()
    inp.cheminformatics = 'openeye'
    inp.nmols = [1]
    inp.symmetry = True
    inp.normalization = False
    inp.resChargeDict = {'meoh':0}
    inp.restraintinfo = {'penalty': '2-stg-fit',
                         'matrices': ['esp'],
                         'a1' : 0.0005, 'a2' : 0.001, 'b'  : 0.1} 
    mol = Molecule_OEMol()
    mol.addInp(inp)
    mol.addCoordFiles('meoh.xyz')
    mol.addEspf('meoh.espf', selectedPts =  [None])

    cal = Respyte_Optimizer()
    cal.addMolecule(mol)
    cal.addInp(inp)
    a1val = inp.restraintinfo['a1']
    a2val = inp.restraintinfo['a2']
    bval  = inp.restraintinfo['b']
    charges, esprrmss, efrrmss = cal.twoStageFit(a1val, a2val, bval) 

    assert [round(q, 4) for q in charges[0]] == [ -0.6498, 0.4215, 0.1166, 0.0372, 0.0372, 0.0372]
    shutil.rmtree('resp_output')
    os.chdir(orig_path)