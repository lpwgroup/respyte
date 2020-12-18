from warnings import warn
import copy
import numpy as np
import sympy
from respyte.molecules import respyte_molecules
from respyte.fbmolecule import PeriodicTable
import respyte.objective

current_equivalence_levels = ['connectivity', 'relaxed_connectivity', 'nosym', 'symbol', 'symbol2']

class respyte_model:
    def __init__(self, model_type, molecules, parameter_types, fix_polar_charges=False, prev_objective=None):
        assert isinstance(molecules, respyte_molecules)
        self.molecules = molecules
        
        self.model_type = model_type

        # read input parameter_types and save the dictionary 
        self.__check_input_parameter_types(parameter_types)

        # get parameter set
        vals, val_info = self.get_parameter_set()
        ls, l_info, combined_fixed_charges = self.get_l_set(fix_polar_charges, prev_objective)
        self.parms = list(vals) + list(ls)
        self.parm_info = list(val_info) + list(l_info)
        self.combined_fixed_charges = combined_fixed_charges 

    def __check_input_parameter_types(self, parameter_types):
        '''
        check if the input parameter_types is valid.
        '''
        raise NotImplementedError("get_parameter_set() should be implemented in subclass")
    def get_parameter_set(self):
        '''
        from the input parameter_types, build parameter set to fit
        '''
        raise NotImplementedError("get_parameter_set() should be implemented in subclass")
    
    def get_l_set(self, fix_polar_charges, prev_objective):
        '''
        add lagrange multipliers for fixed charges 
        '''
        raise NotImplementedError("get_l_set() should be implemented in subclass")

class point_charge_model(respyte_model):
    def __init__(self, model_type, molecules, parameter_types, fix_polar_charges=False, prev_objective=None):
        assert isinstance(molecules, respyte_molecules)
        self.molecules = molecules
        assert model_type in ['point_charge', 'point_charge_numerical']
        self.model_type = model_type

        # read input parameter_types and save the dictionary 
        self.__check_input_parameter_types(parameter_types)

        # get parameter set
        vals, val_info = self.get_parameter_set()
        ls, l_info, combined_fixed_charges = self.get_l_set(fix_polar_charges, prev_objective)
        self.parms = list(vals) + list(ls)
        self.parm_info = list(val_info) + list(l_info)
        self.combined_fixed_charges = combined_fixed_charges 

    def __check_input_parameter_types(self, parameter_types):
        '''
        check if the input parameter_types is valid.
        '''
        assert isinstance(parameter_types, dict)
        if 'charge' in list(parameter_types.keys()):
            param_type = 'charge'
            equiv_level = parameter_types[param_type]
            assert parameter_types[param_type] in current_equivalence_levels, f'charge equivalence level {equiv_level} not supported!'
        else: 
            warn('Charge equivalence level not set. will use "connectivity" by default')
            param_type, equiv_level = 'charge', 'connectivity'
        if list(parameter_types.keys()) !=  ['charge']:
            warn('Redundant parameter types for point charge model were given. remove redundancy and only keep charge equivalence level')
            self.parameter_types = {param_type : equiv_level}
        else:
            self.parameter_types = parameter_types

    def get_parameter_set(self):
        vals = []
        val_info = []
        # using the input equivalence level for charge, get parameter set
        param_type = 'charge'
        equiv_level = self.parameter_types[param_type]
        for equiv in self.molecules.atom_equiv[equiv_level]['equivs']:
            val = 0 # initial guess for atomic partial charges = 0
            vals.append(val) 
            val_info.append([equiv, param_type, equiv_level])
        return vals, val_info

    # def get_l_set(self, fix_polar_charges, prev_objective):
    #     ls, l_info, combined_fixed_charges = self.fix_charges_from_molecules()
    #     if fix_polar_charges and prev_objective is not None:
    #         ls_add, l_info_add, combined_fixed_charges_add = self.fix_polar_charges_from_previous_step(prev_objective, combined_fixed_charges)
    #         ls = list(ls) + list(ls_add)
    #         l_info = list(l_info) + list(l_info_add)
    #         combined_fixed_charges = list(combined_fixed_charges)+ list(combined_fixed_charges_add)
    #     return ls, l_info, combined_fixed_charges

    def get_l_set(self, fix_polar_charges, prev_objective):
        # fix charges from molecules
        param_type = 'charge'
        equiv_level = self.parameter_types[param_type]
        combined_fixed_charges = []
        for mol in self.molecules.mols: 
            for atom_indices, chg in mol.fixed_charges: 
                equivs = mol.convert_index_to_equiv(atom_indices, equiv_level)
                equivs_sorted = sorted(equivs)
                fixed_charge = [equivs_sorted, float(chg)]
                # set min of equivs_sorted to 1 to remove degenerate constraints
                processed_fixed_charge = self.process_fixed_charge(fixed_charge)
                if processed_fixed_charge not in combined_fixed_charges:
                    combined_fixed_charges.append(processed_fixed_charge)
        
        # if fix_polar_charges and prev_objective is not None, add additional fixed polar charges
        if fix_polar_charges and prev_objective is not None:
            assert isinstance(prev_objective, respyte.objective.respyte_objective)
            prev_equiv_level = prev_objective.model.parameter_types[param_type]
            for mol in self.molecules.mols:
                for index in mol.polar_atom_indices:
                    prev_equiv = mol.convert_index_to_equiv(index, prev_equiv_level)
                    equiv = mol.convert_index_to_equiv(index, self.parameter_types[param_type])
                    # get polar charge from the input objective object
                    qidx = prev_objective.parm_info.index([prev_equiv, param_type, prev_equiv_level])
                    chg  = prev_objective.parms[qidx]
                    fixed_charge = [[equiv], float(chg)]
                    if fixed_charge not in combined_fixed_charges: 
                        combined_fixed_charges.append(fixed_charge)

        # remove linear dependencies to avoid singular matrix error
        # 1. gen matrix 
        labels = sorted(list(self.molecules.atom_equiv[equiv_level]['info'].keys()))
        nequivs = len(labels)
        matrix = np.zeros((len(combined_fixed_charges), nequivs))
        for idx, processed_fixed_charge in enumerate(combined_fixed_charges):
            row = np.zeros((nequivs))
            equivs_sorted,chg = processed_fixed_charge
            for equiv in equivs_sorted: 
                equiv_idx = labels.index(equiv)
                row[equiv_idx] += 1
            matrix[idx] = row
        # 2. using sympy rref, find indices of independent rows 
        _, inds = sympy.Matrix(matrix).T.rref()
        if len(inds) != len(combined_fixed_charges):
            print('Warning! There is one or more linearly dependent fixed fragment charges. It will remove redundant fixed fragment charges to avoid singular matrix error!')
        reduced_combined_fixed_charges = []
        ls = []
        l_info = []
        n = 0
        for selected_row_idx in inds: 
            reduced_combined_fixed_charges.append(combined_fixed_charges[selected_row_idx])
            li = 'l' + str(n)
            val = 1 
            ls.append(val)
            l_info.append([li, 'lambda', equiv_level])
            n += 1
        return ls, l_info, reduced_combined_fixed_charges

    def process_fixed_charge(self, fixed_charge):
        occurences_dic = {}
        equivs, chg = copy.deepcopy(fixed_charge)
        unique_equivs = list(set(equivs))
        for equiv in unique_equivs:
            count = equivs.count(equiv)
            occurences_dic[equiv] = count
        gcd = np.gcd.reduce(list(occurences_dic.values()))
        processed_equivs = []
        for equiv in unique_equivs:
            occurences = int(occurences_dic[equiv]/gcd)
            for i in range(occurences):
                processed_equivs.append(equiv)
        processed_chg = float(chg/gcd)
        return  [sorted(processed_equivs), processed_chg]

    # def fix_charges_from_molecules(self): 
    #     param_type = 'charge'
    #     equiv_level = self.parameter_types[param_type]
    #     ls = []
    #     l_info = []
    #     combined_fixed_charges = []
    #     n = 0 
    #     for mol in self.molecules.mols: 
    #         for atom_indices, chg in mol.fixed_charges: 
    #             equivs = mol.convert_index_to_equiv(atom_indices, equiv_level)
    #             equivs_sorted = sorted(equivs)
    #             fixed_charge = [equivs_sorted, float(chg)]
    #             # set min of equivs_sorted to 1 to remove degenerate constraints
    #             processed_fixed_charge = self.process_fixed_charge(fixed_charge)
    #             if processed_fixed_charge not in combined_fixed_charges:
    #                 combined_fixed_charges.append(processed_fixed_charge)
    #                 li = 'l' + str(n)
    #                 val = 1 
    #                 ls.append(val)
    #                 l_info.append([li, 'lambda', equiv_level])
    #                 n += 1
    #     return ls, l_info, combined_fixed_charges
    
    # def fix_polar_charges_from_previous_step(self, prev_objective, current_combined_fixed_charges = None):
    #     # check if the input objective is valid type
    #     assert isinstance(prev_objective, respyte.objective.respyte_objective)
    #     if current_combined_fixed_charges is None: 
    #         current_combiend_fixed_charges = self.combined_fixed_charges
    #     ls_add = []
    #     l_info_add = []
    #     combined_fixed_charges_add = []
    #     param_type = 'charge'
    #     prev_equiv_level = prev_objective.model.parameter_types[param_type]
    #     n = len(current_combined_fixed_charges)
    #     for mol in self.molecules.mols:
    #         for index in mol.polar_atom_indices:
    #             prev_equiv = mol.convert_index_to_equiv(index, prev_equiv_level)
    #             equiv = mol.convert_index_to_equiv(index, self.parameter_types[param_type])
    #             # get polar charge from the input objective object
    #             qidx = prev_objective.parm_info.index([prev_equiv, param_type, prev_equiv_level])
    #             chg  = prev_objective.parms[qidx]
    #             fixed_charge = [[equiv], float(chg)]
    #             if fixed_charge not in current_combined_fixed_charges and fixed_charge not in combined_fixed_charges_add: 
    #                 combined_fixed_charges_add.append(fixed_charge)
    #                 li = 'l' + str(n)
    #                 val = 1
    #                 ls_add.append(val)
    #                 l_info_add.append([li, 'lambda', self.parameter_types[param_type]])
    #                 n += 1
    #     return ls_add, l_info_add, combined_fixed_charges_add

class fuzzy_charge_model(point_charge_model):
    def __init__(self, model_type, molecules, parameter_types, q_core_type='nuclear_charge', alpha0=3 ,fix_polar_charges=False, prev_objective=None):
        assert isinstance(molecules, respyte_molecules)
        self.molecules = molecules
        assert model_type == 'fuzzy_charge'
        self.model_type = model_type

        self.__check_input_parameter_types(parameter_types)

        if isinstance(q_core_type, (int, float)) or q_core_type in ['nuclear_charge', 'valency']:
            self.q_core_type = q_core_type
        else: 
            warn(f'q_core_type {q_core_type} not supported. will set q_core=0 for all atoms by default')
            self.q_core_type = 0 

       # prefactor for alpha0
        assert isinstance(alpha0, (int, float))
        self.alpha0 = float(alpha0)

        # get parameter set
        vals, val_info = self.get_parameter_set()
        ls, l_info, combined_fixed_charges = self.get_l_set(fix_polar_charges, prev_objective)
        self.parms = list(vals) + list(ls)
        self.parm_info = list(val_info) + list(l_info)
        self.combined_fixed_charges = combined_fixed_charges 

    def __check_input_parameter_types(self, parameter_types):
        '''
        check if the input parameter_types is valid.
        '''
        # read input parameter_types and save the dictionary 
        # 1. charge
        assert isinstance(parameter_types, dict)
        if 'charge' in list(parameter_types.keys()):
            param_type = 'charge'
            charge_equiv_level = parameter_types[param_type]
            assert charge_equiv_level in current_equivalence_levels, f'charge equivalence level {charge_equiv_level} not supported!'
        else: 
            warn('Charge equivalence level not set. will use "connectivity" by default')
            param_type, charge_equiv_level = 'charge', 'connectivity'
        # 2. alpha
        if 'alpha' in list(parameter_types.keys()):
            param_type = 'alpha'
            alpha_equiv_level = parameter_types[param_type]
            assert alpha_equiv_level in current_equivalence_levels, f'alpha equivalence level {alpha_equiv_level} not supported!'
        else: 
            warn('Alpha equivalence level not set. will use "symbol" by default')
            param_type, alpha_equiv_level = 'alpha', 'symbol'

        if sorted(list(parameter_types.keys())) != sorted(['charge', 'alpha']):
            warn('Redundant parameter types for fuzzy charge model were given. remove redundancy and only keep charge equivalence level')
            self.parameter_types = {'charge' : charge_equiv_level, 'alpha': alpha_equiv_level}
        else:
            self.parameter_types = parameter_types
            
    def get_q_core_val(self, charge_equiv_level, charge_equiv):
        if isinstance(self.q_core_type, (int, float)):
            q_core = float(self.q_core_type)
        elif self.q_core_type == 'nuclear_charge':
            elem = self.molecules.atom_equiv[charge_equiv_level]['info'][charge_equiv][0]['elem']
            atomic_number = int(list(PeriodicTable.keys()).index(elem) + 1 )
            q_core = float(atomic_number)
        elif self.q_core_type == 'valency':
            elem = self.molecules.atom_equiv[charge_equiv_level]['info'][charge_equiv][0]['elem']
            atomic_number = int(list(PeriodicTable.keys()).index(elem) + 1 )
            if atomic_number < 2+1: 
                q_core = float(atomic_number)
            elif 2 <atomic_number < 10+1:
                q_core = float(atomic_number) - 2
            elif 10 < atomic_number < 18+1:
                q_core = float(atomic_number) - 10
            else:
                raise NotImplementedError('didnt think too much about how to calculate valency:/')
        else:
            raise NotImplementedError
        return q_core

    def get_parameter_set(self):
        vals = []
        val_info = []
        # using the input equivalence level for charge, get parameter set
        param_type = 'charge'
        equiv_level = self.parameter_types[param_type]
        for equiv in self.molecules.atom_equiv[equiv_level]['equivs']:
            val = 0
            vals.append(val) 
            val_info.append([equiv, param_type, equiv_level])
        param_type ='alpha'
        equiv_level = self.parameter_types[param_type]
        for equiv in self.molecules.atom_equiv[equiv_level]['equivs']:
            val = self.alpha0/ float(self.molecules.atom_equiv[equiv_level]['info'][equiv][0]['vdw_radius'])
            vals.append(val)
            val_info.append([equiv, param_type, equiv_level])
        return vals, val_info