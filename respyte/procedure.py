import os, sys, shutil
import copy
from warnings import warn
from respyte.molecules import respyte_molecules
from respyte.objective import respyte_objective
from respyte.optimizer import respyte_optimizer
import scipy.linalg

def single_stage_procedure(molecules, model_type, parameter_types, q_core_type, alpha0, normalize,
                           targets, threshold, penalty, fix_polar_charges=False, prev_objective=None, verbose=True): 
    
    # create objective and add model and penalty function
    objective = respyte_objective(molecules=molecules, normalize=normalize, targets=targets)
    objective.add_model(model_type, parameter_types, q_core_type, alpha0, fix_polar_charges, prev_objective)
    objective.add_penalty(penalty)

    # define an optimizer and run  the optimizer
    optimizer = respyte_optimizer(objective)
    outputs, rrmss = optimizer.run(threshold=threshold, verbose=verbose)

    return  optimizer.objective, outputs, rrmss #, vals
    
penalty_default = {'ptype':'L1', 'a':0.001, 'b':0.1, 'c':0.1}
targets_default = [{'type': 'esp',  'weight': 1.0}]

def resp(molecules, model_type, parameter_types, q_core_type=None, alpha0=None, normalize=False, targets=targets_default, threshold=1e-5, penalty=penalty_default,
         procedure=1, output_path=None, verbose=True):
    if procedure == 2: 
        print('\n\033[1mRunning two-stage fit\033[0m')
        print('1st stage: ')
        mod_parameter_types = copy.deepcopy(parameter_types)
        mod_parameter_types['charge'] = 'relaxed_connectivity'
        objective1, outputs1, rrmss1 = single_stage_procedure(molecules, model_type, mod_parameter_types, q_core_type=q_core_type,
                                            alpha0=alpha0, normalize=normalize, targets=targets, threshold=threshold, penalty=penalty, verbose=verbose)
        # second stage
        print('\n2nd stage: ')
        mod_parameter_types['charge'] = 'connectivity'
        mod_penalty = copy.deepcopy(penalty)
        mod_penalty['a'] *= 2
        objective, outputs, rrmss = single_stage_procedure(molecules, model_type, mod_parameter_types, q_core_type=q_core_type, 
                                            alpha0=alpha0, normalize=normalize, targets=targets, threshold=threshold, penalty=mod_penalty, fix_polar_charges=True, 
                                            prev_objective=objective1, verbose=verbose)
        
    elif procedure == 1: 
        print('\033[1mRunning single-stage fit\033[0m')
        objective, outputs, rrmss = single_stage_procedure(molecules, model_type, parameter_types, q_core_type=q_core_type,
                                            alpha0=alpha0, normalize=normalize, targets=targets, threshold=threshold, penalty=penalty, verbose=verbose)


    if  output_path:
        output_path = os.path.join(output_path, 'resp_outout')
        if os.path.isdir(output_path):
            shutil.rmtree(output_path)
        os.mkdir(output_path)
        write_output(objective, output_path)
    return objective, outputs, rrmss #, vals
 
try:
    import openeye.oechem as oechem
except ImportError:
    warn('The Openeye module not imported. Unable to write output mol2 file.')

def write_output(objective, output_path):    
    if 'openeye.oechem' in sys.modules and objective.model.model_type in ['point_charge', 'point_charge_numerical']:
        for molecule in objective.molecules.mols:
            abspath = molecule.abspath  
            name =  os. path. splitext(os.path.basename(abspath))[0]
            outfile =  os.path.join(output_path, '%s.mol2' %name)  
            oemol = ReadOEMolFromFile(abspath)
            qpot = objective.return_current_values(molecule, 'charge')

            for idx, atom in enumerate(oemol.GetAtoms()):
                atom.SetPartialCharge(qpot[idx])

            ofs = oechem.oemolostream()
            ofs.SetFormat(oechem.OEFormat_MOL2)
            ofs.open(outfile)
            oechem.OEWriteMolecule(ofs, oemol)
    else:  
        print('openeye toolkit is not available or the charge model is not point charge model.\n')
        print('writing output into txt file intead:)')
        for molecule in objective.molecules.mols: 
            abspath = molecule.abspath  
            name =  os.path.splitext(os.path.basename(abspath))[0]
            outfile =  os.path.join(output_path, '%s.txt' %name)  
            output = objective.print_vals_of_single_molecule(molecule,verbose=False)
            with open(outfile, 'w') as outf: 
                for line in output: 
                    outf.write('%s\n'%line)
            outf.close()

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
