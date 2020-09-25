import os, sys, shutil
import copy
from warnings import warn
from respyte.molecules import respyte_molecules
from respyte.objective import respyte_objective
from respyte.optimizer import respyte_optimizer

def single_stage_procedure(molecules, model, penalty, symmetrize='all', prev_objective=None, fix_polar_charges=False, verbose=True):
    assert isinstance(molecules,  respyte_molecules)
    #  create a new molecules object,  re-assign atom ids
    new_molecules = copy.deepcopy(molecules)
    new_molecules.set_atom_id(symmetrize)
    # fix polar charges 
    if fix_polar_charges:
        if prev_objective: 
            new_molecules.fix_polar_charges_from_previous_step(prev_objective)
        else:
            raise RuntimeError('objective object should be provided to fix polar charges!')

    # define an objective function
    objective = respyte_objective(new_molecules, model=model)
    objective.add_penalty(penalty)

    # define an optimizer and run  the optimizer
    optimizer = respyte_optimizer(objective)
    optimizer.run(verbose=verbose)
    return  optimizer.objective

def resp(molecules, symmetry, model, penalty, procedure=1, output_path=None):
    
    if procedure == 2: 
        print('\n\033[1mRunning two-stage fit\033[0m')
        print('1st stage: ')
        if symmetry !=  'all':
            print(f' * You set symmetry={symmetry} for two-stage fit!')
            print('   For the use of canonical two-stage fitting procedure, it will neglect your choice of symmetry.')
        # first stage
        objective1 =  single_stage_procedure(molecules, model, penalty, symmetrize='polar', prev_objective=None, fix_polar_charges=False, verbose=True)
        # second stage
        print('\n2nd stage: ')
        new_penalty =  copy.deepcopy(penalty)
        new_penalty['a'] = penalty['a'] *  2
        objective = single_stage_procedure(molecules, model, new_penalty, symmetrize='all', prev_objective=objective1, fix_polar_charges=True, verbose=True)
    elif procedure == 1: 
        print('\033[1mRunning single-stage fit\033[0m')
        objective = single_stage_procedure(molecules, model, penalty, symmetrize=symmetry, prev_objective=None,  fix_polar_charges=False, verbose=True)
    
    if  output_path:
        output_path = os.path.join(output_path, 'resp_outout')
        if os.path.isdir(output_path):
            shutil.rmtree(output_path)
        os.mkdir(output_path)
        write_output(objective, output_path)
    return objective

try:
    import openeye.oechem as oechem
except ImportError:
    warn('The Openeye module not imported. Unable to write output mol2 file.')

def write_output(objective, output_path):
    if 'openeye.oechem' not in sys.modules:
        print('Can not generate mol2 file without using Openeye toolkit yet.')
    else:
        if objective.model in ['point_charge', 'point_charge_numerical']:
            for molecule in objective.molecules.mols:
                abspath = molecule.abspath  
                name =  os. path. splitext(os.path.basename(abspath))[0]
                outfile =  os.path.join(output_path, '%s.mol2' %name)  
                oemol = ReadOEMolFromFile(abspath)
                qpot = []
                for atomid in molecule.atomids:  
                    qidx = [i[0]  for i in objective.val_info].index(atomid)
                    qpot.append(objective.vals[qidx])
        
                for idx, atom in enumerate(oemol.GetAtoms()):
                    atom.SetPartialCharge(qpot[idx])

                ofs = oechem.oemolostream()
                ofs.SetFormat(oechem.OEFormat_MOL2)
                ofs.open(outfile)
                oechem.OEWriteMolecule(ofs, oemol)
        else:  
            print('Currently  only support point charge model.')
            print('Exits without writing output files.')
        

def ReadOEMolFromFile(filename):
    '''
    Read coordinate file and return a oechem molecule object

            Parameters:
                    filename (str): coordinate file name

            Returns:
                    mol : oechem molecule object
    '''
    ifs = oechem.oemolistream()
    if not ifs.open(filename):
        oechem.OEThrow.Fatal("Cannot open input file %s!" % filename)
    mol = oechem.OEMol()
    if not oechem.OEReadMolecule(ifs, mol):
        oechem.OEThrow.Fatal("Unable to read molecule from %s" % filename)
    ifs.close()
    return mol