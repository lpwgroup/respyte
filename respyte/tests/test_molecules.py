import pytest
import sys, os

from respyte.molecules import *

def test_respyte_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "respyte" in sys.modules

def test_respyte_molecule():
    orig_path = os.getcwd()
    this_file_folder = os.path.dirname(os.path.realpath(__file__))
    test_folder = os.path.join(this_file_folder, 'files')
    test_coord_fnm = os.path.join(test_folder, 'test.pdb')
    test_espf_fnm  = os.path.join(test_folder, 'test.espf')
    molecule  = respyte_molecule(molecule_name='test_mol', coord_fnm=test_coord_fnm, 
                                 espf_fnm=test_espf_fnm, setting=None, input_equiv_atoms=[])
    assert len(set(molecule.atomids))  ==  4
    
    molecule.set_atom_id(symmetry='nosym')
    assert len(set(molecule.atomids))  ==  6

    molecule.set_atom_id(symmetry='polar')
    assert len(set(molecule.atomids))  ==  6

    molecule.set_net_charge(0)
    indices, charge = molecule.fixed_chrages[0]
    assert indices == [0,1,2,3,4,5]
    assert charge == 0

def test_respyte_molecules():
    orig_path = os.getcwd()
    this_file_folder = os.path.dirname(os.path.realpath(__file__))
    test_folder = os.path.join(this_file_folder, 'files')
    test_coord_fnm = os.path.join(test_folder, 'test.pdb')
    test_espf_fnm  = os.path.join(test_folder, 'test.espf')
    molecule  = respyte_molecule(molecule_name='test_mol', coord_fnm=test_coord_fnm, 
                                 espf_fnm=test_espf_fnm, setting=None, input_equiv_atoms=[])
    molecule.set_net_charge(0)

    molecules =  respyte_molecules()
    molecules.add_molecule(molecule)
    molecules.add_molecule(molecule)
    molecules.atomids.sort()
    assert molecules.atomids ==  [0,1,2,3]