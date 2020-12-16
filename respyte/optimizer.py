import numpy as np
import scipy as sci
import copy
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
        self.objective  = copy.deepcopy(objective)

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
                outputs, rrmss, vals = self.objective.print_vals(verbose=verbose)
            else: 
                if verbose: 
                    print(' Iter {:d}. norm(dq): {:.2e} X2: {:.2e}'.format(iteration, ndq, Obj['X']))  
                iteration += 1
                for idx, val in enumerate(self.objective.parms):
                    self.objective.parms[idx] = val + dq[idx]
        return outputs, rrmss, vals
