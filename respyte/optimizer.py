import numpy as np
import scipy as sci
from respyte.objective import respyte_objective

class  respyte_optimizer:
    """ Respyte optimizer class.  

    ...
    Attributes
    ----------
    objective: 
        respyte objective object.  
    
    Methods 
    -------
    run(verbose=True):
        A main respyte  Newton-Raphson optimizer.
    """
    def __init__(self,  objective):
        '''
        Create a respyte_optimizer object.
                Parameters:  
                        objective (respyte_objective): respyte objective object.
        '''
        assert isinstance(objective,  respyte_objective)
        self.objective  = objective 

    def run(self, threshold=1e-8, verbose=True):
        '''
        A main respyte Newton-Raphson  optimizer.
                Parameters: 
                        threshold (float): a convergence criterion of step size
                        verbose (bool): verbose flag
        '''
        converged = False
        iteration = 0
        while converged == False:
            Obj = self.objective.Full()
            dq  = -  sci.linalg.solve(Obj['H'],  Obj['G'])
            # ndq = step size
            ndq  = np.linalg.norm(dq)
            if ndq < threshold: 
                converged = True
                print('\033[1m Converged!\033[0m')
                print('{:>6s} {:>9s} {:>9s} {:>9s} {:>15s} {:>8s} {:8s}'.format('atomid', 'molecule', 'atomname', 'resname', 'model', 'vartype','value'))
                for val, val_info in zip(self.objective.vals, self.objective.val_info):
                    atomid, model, vartype =  val_info
                    if atomid in list(self.objective.molecules.atomid_dict.keys()):
                        atomname = self.objective.molecules.atomid_dict[atomid][0]['atomname']
                        resname  = self.objective.molecules.atomid_dict[atomid][0]['resname']
                        molname = self.objective.molecules.atomid_dict[atomid][0]['molname']
                    else: 
                        atomname = 'None'
                        resname  = 'None'
                        molname = 'None'
                    print('{:>6s} {:>9s} {:>9s} {:>9s} {:>15s} {:>8s} {:>8.4f}'.format(str(atomid), molname, atomname,resname, model, vartype,val))
            else: 
                if verbose: 
                    print(' Iter {:d}. norm(dq): {:.2e}'.format(iteration, ndq))  
                iteration += 1
                for idx, val in enumerate(self.objective.vals):
                    self.objective.vals[idx] = val + dq[idx]
    
        return self.objective
