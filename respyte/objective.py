from warnings import warn
import copy
import numpy as np
from respyte.molecules import respyte_molecules
from respyte.model import respyte_model, point_charge_model, fuzzy_charge_model

# Global variable
bohr2Ang = 0.52918825 # converting Bohr to Angstrom

# X : an objective function
# G : a gradient of the objective function
# H : a hessian of the objective function 
Letters = ['X','G','H']

current_model_types = ['point_charge', 'point_charge_numerical', 'fuzzy_charge']

class esp_target: 
    def __init__(self, molecules, model, normalize):
        self.molecules = copy.deepcopy(molecules) 
        self.model = copy.deepcopy(model)
        self.parm_info = copy.deepcopy(model.parm_info)
        self.normalize = normalize    

    def get(self, vals_):
        Answer = {'X':0.0, 'G':np.zeros(len(vals_)), 'H':np.zeros((len(vals_), len(vals_)))}
        
        for molecule in self.molecules.mols: 
            def compute(vals_):
                '''
                Return a list of residual at a given point, vals_
                '''    
                V = np.zeros((len(molecule.gridxyz)))
                for idx, (gridpt, esp) in enumerate(zip(molecule.gridxyz, molecule.espval)):
                    residual = self.pot_residual(molecule, gridpt, esp, vals_)
                    V[idx] = residual
                return V

            V = compute(vals_)
            Answer['X'] += np.dot(V,V) / (len(V) if self.normalize else 1)

            dV = np.zeros((len(vals_), len(V)))
            if self.model.model_type == 'point_charge': 
                for idx, (gridpt, esp) in enumerate(zip(molecule.gridxyz, molecule.espval)):
                    for atom in molecule.GetAtoms():
                        qidx, q = self.get_val_from_atom(atom, 'charge', vals_)
                        vai,  vai_deriv = single_pt_chg_pot(atom.xyz, q, gridpt)
                        dV[qidx][idx] += -1*vai_deriv
            else: 
                # use a numerical solution of partial differential equations
                for vidx in range(len(vals_)):
                    dV[vidx], _ = f12d3p(fdwrap(compute, vals_, vidx), h=0.0001, f0=V) #need to modify h

            for vidx in range(len(vals_)):
                Answer['G'][vidx] += 2* np.dot(V, dV[vidx,:])  / (len(V) if self.normalize else 1)
                for vidx2 in range(len(vals_)):
                    Answer['H'][vidx, vidx2] += 2 * np.dot(dV[vidx,:],dV[vidx2,:])  / (len(V) if self.normalize else 1)
        
        return Answer

    def get_val_from_atom(self, atom, param_type, vals_):
        equiv_level = self.model.parameter_types[param_type]
        equiv = atom.atom_equiv[equiv_level]
        idx = self.parm_info.index([equiv, param_type, equiv_level])
        val = vals_[idx]
        return idx, val

    def return_current_values(self, molecule, param_type, vals_):
        return_vals = []
        for atom in molecule.GetAtoms():
            vidx, v = self.get_val_from_atom(atom, param_type, vals_)
            return_vals.append(v)
        return return_vals
    
    def pot_residual(self, molecule, gridpt, esp, vals_):
        Vi = 0
        if self.model.model_type in ['point_charge', 'point_charge_numerical']:
            for atom in molecule.GetAtoms():
                qidx, q = self.get_val_from_atom(atom, 'charge', vals_)
                vai,vai_deriv = single_pt_chg_pot(atom.xyz, q, gridpt)
                Vi += vai
        elif self.model.model_type  == 'fuzzy_charge':
            for atom in molecule.GetAtoms():
                qidx, q = self.get_val_from_atom(atom, 'charge', vals_)
                alpha_idx, alpha = self.get_val_from_atom(atom, 'alpha', vals_)

                charge_equiv_level = self.model.parameter_types['charge']
                equiv = atom.atom_equiv[charge_equiv_level]
                q_core = self.model.get_q_core_val(charge_equiv_level, equiv)

                vai = single_fuzzy_chg_pot(atom.xyz, q=q, q_core=q_core, alpha=alpha, gridpt=gridpt)
                Vi += vai
        residual = esp - Vi
        return residual

class ef_target(esp_target):
    def __init__(self, molecules, model, normalize):
        super().__init__(molecules, model, normalize)

    def get(self, vals_):
        Answer = {'X':0.0, 'G':np.zeros(len(vals_)), 'H':np.zeros((len(vals_), len(vals_)))}

        for molecule in self.molecules.mols: 
            def compute(vals_):
                '''
                Return a list of residual at a given point, vals_
                '''    
                V = []
                for idx, (gridpt, ef) in enumerate(zip(molecule.gridxyz, molecule.efvals)):
                    diff, residual = self.field_residual(molecule, gridpt, ef, vals_)
                    V.append(diff) 
                V = np.array(V).flatten()    
                return V      

            V = compute(vals_)
            Answer['X'] += np.dot(V,V) / (len(V) if self.normalize else 1)                  

            dV = np.zeros((len(vals_), len(V)))

            # use a numerical solution of partial differential equations
            for vidx in range(len(vals_)):
                dV[vidx], _ = f12d3p(fdwrap(compute, vals_, vidx), h=0.0001, f0=V) #need to modify h

            for vidx in range(len(vals_)):
                Answer['G'][vidx] += 2* np.dot(V, dV[vidx,:])  / (len(V) if self.normalize else 1)
                for vidx2 in range(len(vals_)):
                    Answer['H'][vidx, vidx2] += 2 * np.dot(dV[vidx,:],dV[vidx2,:])  / (len(V) if self.normalize else 1)
        
        return Answer

    def field_residual(self, molecule,gridpt, ef, vals_):
        Ei = np.zeros(3)
        if self.model.model_type in ['point_charge', 'point_charge_numerical']:
            for atom in molecule.GetAtoms():
                qidx, q = self.get_val_from_atom(atom, 'charge', vals_)
                Eai = single_pt_chg_field(atom.xyz, q, gridpt)
                Ei += Eai
        elif self.model.model_type  == 'fuzzy_charge':
            for atom in molecule.GetAtoms():
                qidx, q = self.get_val_from_atom(atom, 'charge', vals_)
                alpha_idx, alpha = self.get_val_from_atom(atom, 'alpha', vals_)

                charge_equiv_level = self.model.parameter_types['charge']
                equiv = atom.atom_equiv[charge_equiv_level]
                q_core = self.model.get_q_core_val(charge_equiv_level, equiv)

                Eai = single_fuzzy_chg_field(atom.xyz, q=q, q_core=q_core, alpha=alpha, gridpt=gridpt)
                Ei += Eai
        diff = np.array(ef) - np.array(Ei)
        residual = np.linalg.norm(diff)
        # print(f'ef: {ef}, Ei:{Ei}, residual:{residual:.4f}')
        return diff, residual 
    
implemented_targets = {'esp': esp_target, 
                       'ef' : ef_target,
                       }

class respyte_objective: 
    def __init__(self, molecules=None, model=None, normalize=False, targets=None, penalty=None):
        assert isinstance(molecules, respyte_molecules)
        self.molecules = copy.deepcopy(molecules)

        if model: 
            assert isinstance(model, respyte_model)
            self.model = copy.deepcopy(model)
            self.parms = copy.deepcopy(model.parms)
            self.parm_info = copy.deepcopy(model.parm_info)
            self.np = len(self.parms)
            
        assert isinstance(normalize, bool)
        if normalize: 
            print('** Unrestrained obj fn normalized by the number of grid points. **')
        self.normalize = normalize

        self.targets = []
        if targets:
            self.add_targets(targets)            

        if penalty: 
            assert isinstance(penalty, respyte_penalty)
            self.penalty = penalty

    def add_targets(self, targets):
        assert isinstance(targets, list)
        for target in targets: 
            self.add_target(target)

    def add_target(self, target): 
        assert target['type'].lower() in ['esp', 'ef']
        assert isinstance(target['weight'], (int,float))
        # check if the type is already in self.targets 
        exists = False
        for current_target in self.targets: 
            if target['type'].lower() == current_target['type']:
                exists = True
                to_be_popped = current_target
        if exists: 
            print(f'Target {target["type"]} already exists. Ignore the old target and will re-assign the target weight to {target["weight"]}.')
            self.targets.pop(to_be_popped)
        self.targets.append({'type': target['type'].lower(), 'weight': target['weight']})

    def add_model(self, model_type, parameter_types, q_core_type=None, alpha0=None, fix_polar_charges=False, prev_objective=None): 
        assert model_type in current_model_types
        assert isinstance(parameter_types, dict)
        if model_type in ['point_charge', 'point_charge_numerical']:
            model = point_charge_model(model_type=model_type, molecules=self.molecules, parameter_types=parameter_types, 
                                        fix_polar_charges=fix_polar_charges, prev_objective=prev_objective)
        elif model_type == 'fuzzy_charge':
            model = fuzzy_charge_model(model_type=model_type, molecules=self.molecules, parameter_types=parameter_types, 
                                        q_core_type=q_core_type, alpha0=alpha0, fix_polar_charges=fix_polar_charges, prev_objective=prev_objective)
        self.model = copy.deepcopy(model)
        self.parms = copy.deepcopy(model.parms)
        self.parm_info = copy.deepcopy(model.parm_info)
        self.np = len(self.parms)

    def add_penalty(self, penalty_function):
        self.penalty = respyte_penalty(self.molecules, self.model, penalty_function)
    
    def get(self):
        if self.model.model_type in ['point_charge', 'point_charge_numerical', 'fuzzy_charge']:
            return self.get_charge_model()
        else: 
            raise NotImplementedError

    def get_charge_model(self):
        Obj0 = {'X':0.0, 'G':np.zeros(self.np), 'H':np.zeros((self.np, self.np))}

        for tgt in self.targets: 
            tgt_wt = tgt['weight'] 
            target = implemented_targets[tgt['type']](self.molecules, self.model, self.normalize)
            Ans = target.get(self.parms) 
            for i in range(3):
                Obj0[Letters[i]] += Ans[Letters[i]] * tgt_wt

        # charge constraint part 
        Obj1 = {'X':0.0, 'G':np.zeros(self.np), 'H':np.zeros((self.np, self.np))}

        for idx, fixed_charge_info in enumerate(self.model.combined_fixed_charges):
            equivs, fixed_charge  = fixed_charge_info
            lambdai = 'l' + str(idx)
            sum_q = 0
            for equiv in equivs:
                param_type = 'charge'
                equiv_level = self.model.parameter_types[param_type]
                vidx = self.parm_info.index([equiv, param_type, equiv_level])
                sum_q += self.parms[vidx]
            lambdaidx = self.parm_info.index([lambdai, 'lambda', equiv_level])
            vlambda = self.parms[lambdaidx]
            q_diff = sum_q - fixed_charge
            Obj1['X'] += vlambda * q_diff
            Obj1['G'][lambdaidx] = q_diff
            for equiv in equivs:
                vidx = self.parm_info.index([equiv, param_type, equiv_level])
                Obj1['G'][vidx] += vlambda
                Obj1['H'][vidx][lambdaidx] += 1
                Obj1['H'][lambdaidx][vidx] += 1
            
        Answer = {'X':0.0, 'G':np.zeros(self.np), 'H':np.zeros((self.np, self.np))}
        for i in range(3):
            Answer[Letters[i]] = Obj0[Letters[i]] + Obj1[Letters[i]]

        return Answer
       
    def get_val_from_atom(self, atom, param_type, vals_):
        equiv_level = self.model.parameter_types[param_type]
        equiv = atom.atom_equiv[equiv_level]
        idx = self.parm_info.index([equiv, param_type, equiv_level])
        val = vals_[idx]
        return idx, val

    def return_current_values(self, molecule, param_type):
        vals_ = []
        for atom in molecule.GetAtoms():
            vidx, v = self.get_val_from_atom(atom, param_type, self.parms)
            vals_.append(v)
        return vals_
    
    def pot_residual(self, molecule, gridpt, esp, vals_):
        Vi = 0
        if self.model.model_type in ['point_charge', 'point_charge_numerical']:
            for atom in molecule.GetAtoms():
                qidx, q = self.get_val_from_atom(atom, 'charge', vals_)
                vai,vai_deriv = single_pt_chg_pot(atom.xyz, q, gridpt)
                Vi += vai
        elif self.model.model_type  == 'fuzzy_charge':
            for atom in molecule.GetAtoms():
                qidx, q = self.get_val_from_atom(atom, 'charge', vals_)
                alpha_idx, alpha = self.get_val_from_atom(atom, 'alpha', vals_)

                charge_equiv_level = self.model.parameter_types['charge']
                equiv = atom.atom_equiv[charge_equiv_level]
                q_core = self.model.get_q_core_val(charge_equiv_level, equiv)

                vai = single_fuzzy_chg_pot(atom.xyz, q=q, q_core=q_core, alpha=alpha, gridpt=gridpt)
                Vi += vai
        residual = esp - Vi
        return residual

    # def field_residual(self, molecule,gridpt, ef, vals_):
    #     Ei = 0
    #     if self.model.model_type in ['point_charge', 'point_charge_numerical']:
    #         for atom in molecule.GetAtoms():
    #             qidx, q = self.get_val_from_atom(atom, 'charge', vals_)
    #             Eai = single_pt_chg_field(atom.xyz, q, gridpt)
    #             Ei += Eai
    #     elif self.model.model_type  == 'fuzzy_charge':
    #         for atom in molecule.GetAtoms():
    #             qidx, q = self.get_val_from_atom(atom, 'charge', vals_)
    #             alpha_idx, alpha = self.get_val_from_atom(atom, 'alpha', vals_)

    #             charge_equiv_level = self.model.parameter_types['charge']
    #             equiv = atom.atom_equiv[charge_equiv_level]
    #             q_core = self.model.get_q_core_val(charge_equiv_level, equiv)

    #             Eai = single_fuzzy_chg_field(atom.xyz, q=q, q_core=q_core, alpha=alpha, gridpt=gridpt)
    #             Ei += Eai
    #     diff = np.array(ef) - np.array(Ei)
    #     residual = np.linalg.norm(diff)
    #     return residual        

    def Full(self):
        '''
        calculate an objective function with penalty contribution.
        '''
        Objective = self.get()
        Objective['X0'] = Objective['X']
        Objective['G0'] = Objective['G'].copy()
        Objective['H0'] = Objective['H'].copy()
        # add penalty contribution
        if self.penalty :
            Extra = self.penalty.get(self.parms) 
            for i in range(3):
                Objective[Letters[i]] += Extra[i]
        return Objective

    def return_esp_rrms(self, molecule):
        residuals = np.zeros((len(molecule.gridxyz)))
        for idx, (gridpt, esp) in enumerate(zip(molecule.gridxyz, molecule.espval)):
            residual = self.pot_residual(molecule, gridpt, esp, self.parms)
            residuals[idx] = residual
        chi_esp_sq = np.sum(np.dot(np.array(residuals), np.array(residuals)))       
        esp_sq = np.sum(np.dot(np.array(molecule.espval),np.array(molecule.espval)))
        rrmsval = np.sqrt(chi_esp_sq/esp_sq)
        return rrmsval

    def print_vals(self, verbose=True): 
        outputs= []
        rrmss = []
        vals = []
        for molecule in self.molecules.mols: 
            output,rrms, data = self.print_vals_of_single_molecule(molecule, verbose)
            outputs.append(output)
            rrmss.append(rrms)
            vals.append(data)
        return outputs, rrmss, vals
  
    def print_vals_of_single_molecule(self, molecule, verbose=True):
        output = []
        rrms = self.return_esp_rrms(molecule)
        head1 = f'mol: {molecule.name}, model: {self.model.model_type}'
        head2 = 'RRMS: {:>9.4f}'.format(rrms)
        output.append(head1)
        output.append(head2)
        if verbose:
            print(head1)
            print(head2)
        header = '{:>6s} {:>9s} {:>9s} {:>9s} {:>11s}' + ' {:>9s}'*len(list(self.model.parameter_types.keys())) 
        body = '{:>6s} {:>9.4f} {:>9.4f} {:>9.4f} {:>11s}' + ' {:>9.4f}'*len(list(self.model.parameter_types.keys()))
        head3 = header.format('elem', 'x', 'y', 'z', 'equiv(chg)', *list(self.model.parameter_types.keys()))
        output.append(head3)
        if verbose:
            print(head3)
        data = {}
        for param_type, equiv_level in self.model.parameter_types.items():
            vals = self.return_current_values(molecule, param_type)
            data[param_type] = vals

        for atom in molecule.GetAtoms():
            equiv = atom.atom_equiv[self.model.parameter_types['charge']]
            vals = []
            for param_type, vals_ in data.items():
                vals.append(vals_[atom.idx])
            line = body.format(atom.elem, atom.xyz[0], atom.xyz[1], atom.xyz[2], str(equiv), *vals)
            output.append(line)
            if verbose:
                print(line)
        return output, rrms, data

class respyte_penalty: 
    def __init__(self, molecules, model, penalty_function):
        self.molecules = molecules
        self.model = model
        assert isinstance(penalty_function, dict)
        self.penalty_function = penalty_function

    def get(self, vals_):
        if self.model.model_type in ['point_charge', 'point_charge_numerical']:
            return self.get_point_charge(vals_)
        elif self.model.model_type == 'fuzzy_charge':
            return self.get_fuzzy_charge(vals_)
        else: 
            raise NotImplementedError

    def get_point_charge(self, vals_):
        X2 = 0
        G2 = np.zeros((len(vals_)))
        H2 = np.zeros((len(vals_), len(vals_)))
        if self.penalty_function['ptype'] == 'L1':
            # L1: hyperbolic penalty function
            if 'a' in self.penalty_function.keys():
                a = self.penalty_function['a'] 
            else: 
                warn('a(prefac of charge regularization) not set. will use default a=0.001')
                a = 0.001 
            if 'b' in self.penalty_function.keys():
                b = self.penalty_function['b']
            else:
                warn('b(tightness of charge regularization) not set. will use default b=0.1')
                b = 0.1

            for molecule in self.molecules.mols:
                param_type = 'charge'
                equiv_level = self.model.parameter_types[param_type]
                for equiv  in molecule.atom_equiv[equiv_level]['equivs']:
                    if molecule.atom_equiv[equiv_level]['info'][equiv][0]['elem'] != 'H':
                        vidx = self.model.parm_info.index([equiv, param_type, equiv_level])
                        q = vals_[vidx]
                        x = a * (np.sqrt(q**2 + b**2) - b)
                        g = a * q / np.sqrt(q**2 + b**2)
                        h = a * b**2 / np.power(q**2 + b**2, 3/2)

                        X2 += x
                        G2[vidx] += g
                        H2[vidx][vidx] += h
        if self.penalty_function['ptype'] == 'L2':
            # L1: harmonic penalty function
            if 'a' in self.penalty_function.keys():
                a = self.penalty_function['a'] 
            else: 
                warn('a(prefac of charge regularization) not set. will use default a=0.001')
                a = 0.001 
            for molecule in self.molecules.mols:
                param_type = 'charge'
                equiv_level = self.model.parameter_types[param_type]
                for equiv  in molecule.atom_equiv[equiv_level]['equivs']:
                    if molecule.atom_equiv[equiv_level]['info'][equiv][0]['elem'] != 'H':
                        vidx = self.model.parm_info.index([equiv, param_type, equiv_level])
                        q = vals_[vidx]
                        x = a * q**2 
                        g = 2*a * q
                        h = 2*a

                        X2 += x
                        G2[vidx] += g
                        H2[vidx][vidx] += h
        if self.penalty_function['ptype'] == 'Hinv': 
            raise NotImplementedError

        return X2, G2, H2


    def get_fuzzy_charge(self, vals_):

        X2, G2, H2 = self.get_point_charge(vals_)

        if 'c'  in self.penalty_function.keys():
            c = self.penalty_function['c']
        else:
            warn('c( prefac of alpha regularization) not set. will use default c=0.1')
            c = 0.1
        for molecule in self.molecules.mols:
            param_type = 'alpha'
            equiv_level = self.model.parameter_types[param_type]
            for equiv  in molecule.atom_equiv[equiv_level]['equivs']:
                alpha_idx = self.model.parm_info.index([equiv, param_type, equiv_level])
                alpha = vals_[alpha_idx]
                alpha0 = self.model.alpha0/ float(molecule.atom_equiv[equiv_level]['info'][equiv][0]['vdw_radius'])
                x = c * (alpha - alpha0)**2
                g = 2 * c * (alpha - alpha0)
                h = 2 * c

                X2 += x
                G2[alpha_idx] += g
                H2[alpha_idx][alpha_idx] += h

        return X2, G2, H2

def single_pt_chg_pot(xyz, q, gridpt):
    '''
    Return a potential at a grid point due to a point charge q at xyz. 

            Parameters:
                    xyz (list): xyz coordinates of a point charge (Angstrom)
                    q (float): a point charge 
                    gridpt (list): xyz coordinates of a grid point (Angstrom)

            Returns:
                    pot (float): potential at a grid point
                    pot_deriv (float): derivative of a potential w.r.t a point charge
    '''
    xyz = np.array(xyz) /bohr2Ang
    gridpt = np.array(gridpt) /bohr2Ang
    dist = np.sqrt(np.dot(xyz-gridpt, xyz-gridpt))
    pot = q/dist
    pot_deriv = 1/dist
    return pot, pot_deriv

def single_pt_chg_field(xyz, q, gridpt):
    '''
    Return a electric field at a grid point due to a point charge q at xyz. 
            Parameters:
                    xyz (list): xyz coordinates of a point charge (Angstrom)
                    q (float): a point charge 
                    gridpt (list): xyz coordinates of a grid point (Angstrom)
            Returns:
                    field (float): electric field at a grid point
    '''
    xyz = np.array(xyz) /bohr2Ang
    gridpt = np.array(gridpt) /bohr2Ang
    dist = np.sqrt(np.dot(xyz-gridpt, xyz-gridpt))
    prefactor = q / pow(dist, 3)
    field = (gridpt - xyz) * prefactor ##
    return field

def single_fuzzy_chg_pot(xyz, q, q_core, alpha, gridpt):
    '''
    Return a potential at a grid point due to a fuzzy charge at xyz. 

            Parameters:
                    xyz (list): xyz coordinates of a point charge (Angstrom)
                    q (float): overall partial charge of a fuzzy charge
                    q_core (float): core charge of a fuzzy charge
                    alpha (float): smearing parameter of a fuzzy charge (1/Angstrom)
                    gridpt (list): xyz coordinates of a grid point (Angstrom)

            Returns:
                    pot (float): potential at a grid point
    '''
    xyz = np.array(xyz) /bohr2Ang
    gridpt = np.array(gridpt) /bohr2Ang
    alpha_bohr = alpha * bohr2Ang
    dist = np.sqrt(np.dot(xyz-gridpt, xyz-gridpt))
    pot = q_core/dist + (q-q_core)/dist * (1-np.exp(-1* alpha*dist))
    return pot

def single_fuzzy_chg_field(xyz, q, q_core, alpha, gridpt):
    '''
    Return a field at a grid point due to a fuzzy charge at xyz. 
            Parameters:
                    xyz (list): xyz coordinates of a point charge (Angstrom)
                    q (float): overall partial charge of a fuzzy charge
                    q_core (float): core charge of a fuzzy charge
                    alpha (float): smearing parameter of a fuzzy charge (1/Angstrom)
                    gridpt (list): xyz coordinates of a grid point (Angstrom)
            Returns:
                    field (float): electric field at a grid point
    '''
    xyz = np.array(xyz) /bohr2Ang
    gridpt = np.array(gridpt) /bohr2Ang
    alpha_bohr = alpha * bohr2Ang
    dist = np.sqrt(np.dot(xyz-gridpt, xyz-gridpt))
    dist2 = pow(dist, 2)
    dist3 = pow(dist, 3)
    dampf = 1-np.exp(-1* alpha*dist)
    prefactor = q_core/dist3 + (q-q_core)/dist3 * dampf - (q-q_core) * alpha_bohr * (1-dampf)/dist2 
    field = (gridpt - xyz) * prefactor ##
    return field

# finite differences (copy-pasted from ForceBalance source code)
def fdwrap(func,mvals0,pidx,key=None,**kwargs):
    def func1(arg):
        mvals = list(mvals0)
        mvals[pidx] += arg
        if key is not None:
            return func(mvals,**kwargs)[key]
        else:
            return func(mvals,**kwargs)
    return func1

def f12d3p(f, h, f0 = None):
    if f0 is None:
        fm1, f0, f1 = [f(i*h) for i in [-1, 0, 1]]
    else:
        fm1, f1 = [f(i*h) for i in [-1, 1]]
    fp = (f1-fm1)/(2*h)
    fpp = (fm1-2*f0+f1)/(h*h)
    return fp, fpp



