
import numpy as np


def single_pt_chg_pot(xyz, q, gridpt):
    '''
    Return a potential at a grid point due to a point charge q at xyz. 

            Parameters:
                    xyz (list): xyz coordinates of a point charge 
                    q (float): a point charge 
                    gridpt (list): xyz coordinates of a grid point

            Returns:
                    pot (float): potential at a grid point
                    pot_deriv (float): derivative of a potential w.r.t a point charge
    '''
    xyz = np.array(xyz)
    gridpt = np.array(gridpt)
    dist = np.sqrt(np.dot(xyz-gridpt, xyz-gridpt))
    pot = q/dist
    pot_deriv = 1/dist
    return pot, pot_deriv

def single_fuzzy_chg_pot(xyz, q, q_core, alpha, gridpt):
    '''
    Return a potential at a grid point due to a fuzzy charge at xyz. 

            Parameters:
                    xyz (list): xyz coordinates of a point charge 
                    q (float): overall partial charge of a fuzzy charge
                    q_core (float): core charge of a fuzzy charge
                    alpha (float): smearing parameter of a fuzzy charge
                    gridpt (list): xyz coordinates of a grid point

            Returns:
                    pot (float): potential at a grid point
    '''
    xyz = np.array(xyz)
    gridpt = np.array(gridpt)
    dist = np.sqrt(np.dot(xyz-gridpt, xyz-gridpt))
    pot = q_core/dist + (q-q_core)/dist * (1-np.exp(-1* alpha*dist))
    return pot

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

# X : an objective function
# G : a gradient of the objective function
# H : a hessian of the objective function 
Letters = ['X','G','H']

class respyte_objective: 
    """ Respyte objective class.  

    ...
    Attributes
    ----------
    molecules: respyte_molecules
        respyte molecules object.  
    model: str
        type of model, currently support "point_charge", "point_charge_numerical" and "fuzzy_charge" 
    vals: list
        list of values
    val_info: list
        list of informations([atom id, model, vartype])
    nv: int
        number of variables
    penalty: respyte_penalty
        penalty object
    
    Methods 
    -------
    add_penalty(penalty_type):
        add penalty object
    get:
        calculate an objective function with charge constraints  
    Full: 
        calculate an objective function with penalty contribution
    """
    def __init__(self, molecules=None, model='point_charge'):
        self.molecules = molecules
        self.model = model
        self.vals = []
        self.val_info = []
        # print(f'model:{self.model}')
        # determine len of vals(=degrees of freedom) from atom id list of input molecules 
        if self.model in ['point_charge', 'point_charge_numerical']:
            for atomid  in self.molecules.atomids: 
                chg = 0
                self.vals.append(chg)
                self.val_info.append([atomid, self.model, 'q'])
            for idx, fixed_charge in enumerate(self.molecules.fixed_charges):
                lambdai = 'l' + str(idx)
                self.vals.append(1)
                self.val_info.append([lambdai, self.model, 'lambda'])

        elif self.model == 'fuzzy_charge':
            for atomid in self.molecules.atomids: 
                self.vals.append(0)
                self.val_info.append([atomid, self.model, 'q'])
                alpha = 3 / float(self.molecules.atomid_dict[atomid][0]['vdw_radius'])
                self.vals.append(alpha)
                self.val_info.append([atomid, self.model, 'alpha'])
            for idx, fixed_charge in enumerate(self.molecules.fixed_charges):
                lambdai = 'l' + str(idx)
                self.vals.append(1)
                self.val_info.append([lambdai, self.model, 'lambda'])
        # print(f"val_info: {self.val_info}")
        self.nv = len(self.vals)
        self.penalty = None

    def add_penalty(self, penalty_type):
        self.penalty = respyte_penalty(self.molecules, self.val_info, self.model, penalty_type)

    def get(self):
        '''
        calculate objective function with charge constraints.
        '''
        X0=0
        G0 = np.zeros(self.nv)
        H0 = np.zeros((self.nv, self.nv))
        def compute(vals_):
            '''
            Return a list of residual at a given point, vals_
            '''
            V = []
            for molecule in self.molecules.mols: 
                residuals = np.zeros((len(molecule.gridxyz)))
                for idx, (gridpt, esp) in enumerate(zip(molecule.gridxyz, molecule.espval)):
                    # potential on a grid point
                    Vi = 0
                    for atom in molecule.GetAtoms():           
                        qidx= self.val_info.index([atom.id, self.model, 'q'])
                        q = vals_[qidx]
                        # vai = potential contribution of an atom
                        if self.model in ['point_charge', 'point_charge_numerical']:
                            vai,vai_deriv = single_pt_chg_pot(atom.xyz, q, gridpt)
                        elif self.model == 'fuzzy_charge':
                            alpha_idx = self.val_info.index([atom.id, self.model, 'alpha'])
                            alpha = vals_[alpha_idx]
                            vai = single_fuzzy_chg_pot(atom.xyz, q=q, q_core=0, alpha=alpha, gridpt=gridpt)
                        Vi += vai
                    residual = esp - Vi
                    residuals[idx] = residual
                residuals_list = residuals.tolist()
                V += residuals_list
            return np.array(V)

        V = compute(self.vals)
        X0 = np.dot(V,V)
        dV = np.zeros((self.nv, len(V)))

        if self.model == 'point_charge':
            # use an analytic solution of partial differential equations
            loc = 0
            for molecule in self.molecules.mols:
                for idx, (gridpt, esp) in enumerate(zip(molecule.gridxyz, molecule.espval)):
                    for atom in molecule.GetAtoms():                 
                        qidx= self.val_info.index([atom.id, self.model, 'q'])
                        q = self.vals[qidx]
                        vai,  vai_deriv = single_pt_chg_pot(atom.xyz, q, gridpt)
                        
                        dV[qidx][idx+loc] += -1*vai_deriv
                loc  += len(molecule.gridxyz)
        else: 
            # use a numerical solution of partial differential equations
            for vidx in range(self.nv):
                dV[vidx], _ = f12d3p(fdwrap(compute, self.vals, vidx), h=0.0001, f0=V) #need to modify h

        for vidx in range(self.nv):
            G0[vidx] = 2* np.dot(V, dV[vidx,:])
            for vidx2 in range(self.nv):
                H0[vidx, vidx2] = 2 * np.dot(dV[vidx,:],dV[vidx2,:])

        # charge constraint part 
        X1=0
        G1 = np.zeros(self.nv)
        H1 = np.zeros((self.nv, self.nv))
        for idx, fixed_charge_info in  enumerate(self.molecules.fixed_charges):
            list_of_ids, fixed_charge  = fixed_charge_info
            lambdai = 'l' + str(idx)
            sum_q = 0
            for atomid in list_of_ids:
                vidx= self.val_info.index([atomid, self.model, 'q'])
                sum_q += self.vals[vidx]
            lambdaidx = self.val_info.index([lambdai, self.model, 'lambda'])
            vlambda = self.vals[lambdaidx]
            q_diff = sum_q - fixed_charge
            X1 += vlambda * q_diff
            G1[lambdaidx] = q_diff
            for atomid in list_of_ids:
                vidx= self.val_info.index([atomid, self.model, 'q'])
                G1[vidx] += vlambda
                H1[vidx][lambdaidx] += 1
                H1[lambdaidx][vidx] += 1

        X = X0 + X1
        G = G0 + G1
        H = H0 + H1
        Answer = {'X':X, 'G':G, 'H':H}
        return Answer

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
            Extra = self.penalty.get(self.vals) 
            for i in range(3):
                Objective[Letters[i]] += Extra[i]
        return Objective


class respyte_penalty:
    """ Respyte penalty object 

    ...
    Attributes
    ----------
    molecules: respyte_molecules
        respyte molecules object.  
    model: str
        type of model, currently support "point_charge", "point_charge_numerical" and "fuzzy_charge" 
    val_info: list
        list of informations([atom id, model, vartype])
    penalty: respyte_penalty
        penalty object
    
    Methods 
    -------
    """
    def __init__(self, molecules, val_info, model, penalty_type):
        self.molecules = molecules
        self.val_info = val_info
        self.penalty_type = penalty_type # dict
        self.model = model

    def get(self, vals_):
        '''
        calculate penalty contribution.
        '''
        X2 = 0
        G2 = np.zeros((len(vals_)))
        H2 = np.zeros((len(vals_), len(vals_)))
        if self.penalty_type['type'] == 'L2':
            a = self.penalty_type['a'] 
            b = self.penalty_type['b']
            for molecule in self.molecules.mols:
                for atomid in molecule.atomids:
                    if self.molecules.atomid_dict[atomid][0]['elem'] != 'H':
                        vidx = self.val_info.index([atomid, self.model, 'q'])
                        q = vals_[vidx]
                        x = a * (np.sqrt(q**2 + b**2) - b)
                        g = a * q / np.sqrt(q**2 + b**2)
                        h = a * b**2 / np.power(q**2 + b**2, 3/2)

                        X2 += x
                        G2[vidx] += g
                        H2[vidx][vidx] += h

        if  self.model == 'fuzzy_charge':
            if 'c'  in self.penalty_type.keys():
                c = self.penalty_type['c']
            else:
                c = 0.1
            for molecule in self.molecules.mols:
                for atomid in molecule.atomids:
                    alpha_idx = self.val_info.index([atomid, self.model, 'alpha'])
                    alpha = vals_[alpha_idx]
                    alpha0 = 3 / self.molecules.atomid_dict[atomid][0]['vdw_radius']
                    x = c * (alpha - alpha0)**2
                    g = 2 * c * (alpha - alpha0)
                    h = 2 * c

                    X2 += x
                    G2[alpha_idx] += g
                    H2[alpha_idx][alpha_idx] += h
        return X2, G2, H2








            

