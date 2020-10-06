import pytest
import sys, os

from respyte.molecules import *
from respyte.parse import *

def test_respyte_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "respyte" in sys.modules

orig_path = os.getcwd()
this_file_folder = os.path.dirname(os.path.realpath(__file__))
test_folder = os.path.join(this_file_folder, 'files')

def test_respyte_molecule():
    test_coord_fnm = os.path.join(test_folder, 'test.mol2')
    test_espf_fnm  = os.path.join(test_folder, 'test.espf')
    molecule  = respyte_molecule(molecule_name='meoh', coord_fnm=test_coord_fnm, 
                                 espf_fnm=test_espf_fnm, settings=None, input_equiv_atoms=[])

    # check atom_equiv
    assert molecule.atom_equiv['nosym']['equivs'] == [0, 1, 2, 3, 4, 5]
    assert molecule.atom_equiv['connectivity']['equivs'] == [2, 0, 3, 1, 1, 1]
    assert molecule.atom_equiv['relaxed_connectivity']['equivs'] == [2, 0, 3, 4, 5, 6]
    assert molecule.atom_equiv['symbol']['equivs'] == [8, 1, 6, 1, 1, 1]
    assert molecule.atom_equiv['symbol2']['equivs'] == [8, -1, 6, 1, 1, 1]

    # polar atom indices and polar hydrogen indices
    assert molecule.polar_atom_indices == [0,1]
    assert molecule.polar_hydrogen_indices == [1]
    
    # net charge
    molecule.set_net_charge(0)
    assert molecule.fixed_charges == [[[0, 1, 2, 3, 4, 5], 0]]

    # 
    invalid_input_equiv_atoms = [[['invalid_name'], ['C1', 'H2'], ['MOL']]]
    molecule.add_input_equiv_atoms(invalid_input_equiv_atoms)
    assert molecule.atom_equiv['nosym']['equivs'] == [0, 1, 2, 3, 4, 5]
    assert molecule.atom_equiv['connectivity']['equivs'] == [2, 0, 3, 1, 1, 1]
    assert molecule.atom_equiv['relaxed_connectivity']['equivs'] == [2, 0, 3, 4, 5, 6]
    assert molecule.atom_equiv['symbol']['equivs'] == [8, 1, 6, 1, 1, 1]
    assert molecule.atom_equiv['symbol2']['equivs'] == [8, -1, 6, 1, 1, 1]

    input_equiv_atoms = [[['meoh'], ['C1', 'H2'], ['MOL']]]
    molecule.add_input_equiv_atoms(input_equiv_atoms)
    assert molecule.atom_equiv['nosym']['equivs'] == [0, 1, 2, 2, 4, 5]
    assert molecule.atom_equiv['connectivity']['equivs'] == [2, 0, 1, 1, 1, 1]
    assert molecule.atom_equiv['relaxed_connectivity']['equivs'] == [2, 0, 3, 3, 5, 6]
    assert molecule.atom_equiv['symbol']['equivs'] == [8, 1, 6, 1, 1, 1]
    assert molecule.atom_equiv['symbol2']['equivs'] == [8, -1, 6, 1, 1, 1]

def test_respyte_molecules():
    test_coord_fnm = os.path.join(test_folder, 'test.mol2')
    test_espf_fnm  = os.path.join(test_folder, 'test.espf')
    molecule  = respyte_molecule(molecule_name='meoh', coord_fnm=test_coord_fnm, 
                                 espf_fnm=test_espf_fnm, settings=None, input_equiv_atoms=[])
    molecule.set_net_charge(0)

    molecules =  respyte_molecules()
    molecules.add_molecule(molecule)
    molecules.add_molecule(molecule)
    
    assert molecules.atom_equiv['nosym']['equivs'] == [0, 1, 2, 3, 4, 5]
    assert molecules.atom_equiv['relaxed_connectivity']['equivs'] == [0, 2, 3, 4, 5, 6]
    assert molecules.atom_equiv['connectivity']['equivs'] == [0, 1, 2, 3]
    assert molecules.atom_equiv['symbol']['equivs'] == [1, 6, 8]
    assert molecules.atom_equiv['symbol2']['equivs'] == [-1, 1, 6, 8]

def test_from_input():
    input_file = os.path.join(test_folder, 'input','respyte_for_test.yml')
    inp = Input(input_file)
    molecules = respyte_molecules()
    molecules.from_input(inp)

    assert molecules.mols[0].fixed_charges == [[[0, 1, 2, 3, 4, 5], 0], [[2], 0.1]]
    assert molecules.mols[0].atom_equiv['nosym']['equivs'] == [0, 1, 2, 2, 4, 5]
    assert molecules.atom_equiv['nosym']['equivs'] == [0, 1, 2, 4, 5]





