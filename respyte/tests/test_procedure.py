import pytest
import sys, os

from respyte.parse import *
from respyte.molecules import *
from respyte.procedure import *

orig_path = os.getcwd()
this_file_folder = os.path.dirname(os.path.realpath(__file__))
test_folder = os.path.join(this_file_folder, 'files')

def test_point_charge_fitting():

    input_file = os.path.join(test_folder, 'input','respyte.yml')
    inp = Input(input_file)

    molecules = respyte_molecules()
    molecules.from_input(inp)

    inp.procedure = 1
    objective = resp(molecules, inp.model_type, inp.parameter_types, inp.q_core_type, inp.alpha0,
            inp.penalty, inp.procedure, verbose=True)
    assert objective.parm_info == [[0, 'charge', 'connectivity'],
                                   [1, 'charge', 'connectivity'],
                                   [2, 'charge', 'connectivity'],
                                   [3, 'charge', 'connectivity'],
                                   ['l0', 'lambda', 'connectivity']]
    assert objective.parms == [0.38616005723158114,
                               0.05045220155142483,
                              -0.592211419650367,
                               0.05469475776451148,
                               0.024659277282952564]
    
    inp.procedure = 2
    objective = resp(molecules, inp.model_type, inp.parameter_types, inp.q_core_type, inp.alpha0,
            inp.penalty, inp.procedure, verbose=True)
    assert objective.parms == [0.42155252290088474,
                               0.0372125686491922,
                              -0.6498465415844541,
                               0.11665631273599276,
                               0.028493381477486823,
                               0.013295609199715976,
                              -0.026121796851129555]
    assert objective.parms

