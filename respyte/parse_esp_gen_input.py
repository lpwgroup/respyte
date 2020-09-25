import os, sys
import yaml
from warnings import warn
import numpy as np
from collections import OrderedDict, namedtuple, Counter

class Input:
    def __init__(self, inputFile=None):
        if inputFile is None:
            self.mols = {}
            self.settings = {}
            self.gridOptions = {}
        else:
            self.readinp(inputFile)
            self.gridOptions = self.genGridOptions(self.settings)
            print(f'  * gridOptions: {self.gridOptions}')


    def readinp(self, inputfile):
        print(f'\033[1m Parse input file, {os.path.abspath(inputfile)}: \033[0m')

        inp = yaml.load(open(inputfile), yaml.SafeLoader)
        self.inp_dir = os.path.dirname(os.path.abspath(inputfile))

        """
        inp.mols = dict(
            mol1 : {'nconf' : 10, 'net_charge': 0.0}, 
            mol2 : {'nconf' : 10, 'net_charge': 1.0}, ...)
        """
        mols = OrderedDict()
        if 'molecules' in inp:
            for mol in inp['molecules']:
                mols[mol] ={'nconf': int(inp['molecules'][mol])}
                if 'charges' in inp:
                    if mol in inp['charges']:
                        net_chg = inp['charges'][mol]
                        assert isinstance(net_chg, (int, float)), f'charge should be a number. charge: {net_chg} is given for molecule,{mol}' 
                    else: 
                        net_chg = 0
                else: 
                    net_chg = 0
                mols[mol]['net_charge'] = net_chg

        if 'settings' in inp:
            settings = inp['settings']
        else:
            # If grid_setting is not provided, use default setting.
            settings = {'type'   : 'msk',
                        'radii'  : 'bondi',
                        'method' : 'hf',
                        'basis'  : '6-31g*',
                        'pcm'    : 'N'}

        print(f'  * mols: {mols}')
        print(f'  * settings: {settings}')
        self.mols     = mols
        self.settings = settings

    def genGridOptions(self, settings):
        gridOptions = {}
        # starts with default settings
        gridOptions['radii'] = 'bondi'
        gridOptions['space'] = 0.7
        gridOptions['inner'] = 1.4
        gridOptions['outer'] = 2.0
        gridType = settings['type']
        if gridType == 'extendedmsk' or gridType == 'extendedMsk':
            gridOptions['gridType'] = 'extendedMsk'
            gridOptions['space'] = 1.0
            gridOptions['inner'] = 0.8
            gridOptions['outer'] = 2.4
        elif gridType=='MSK' or gridType=='msk':
            gridOptions['gridType'] = 'MSK'
            gridOptions['space'] = 1.0
        elif gridType == 'newfcc':
            gridOptions['gridType'] = 'shellFac'
            gridOptions['inner'] = 0.8
            gridOptions['outer'] = 2.4
        elif gridType=='fcc':
            gridOptions['gridType'] = 'shellFacConst'
            gridOptions['outer'] = 1.0
        elif gridType=='vdwfactors':
            gridOptions['gridType'] = 'shellFac'
        elif gridType=='vdwconstants':
            gridOptions['gridType'] = 'shellConst'
            gridOptions['inner'] = 0.4
            gridOptions['outer'] = 1.0
        else:
            raise RuntimeError('unrecognized grid type %s' % gridType)

        if 'radii' in settings:
            gridOptions['radii'] = settings['radii']
        if 'inner' in settings:
            gridOptions['inner'] = settings['inner']
        if 'outer' in settings:
            gridOptions['outer'] = settings['outer']
        if 'space' in settings:
            gridOptions['space'] = settings['space']
        return gridOptions

def main():
    inp = Input(sys.argv[1])
if __name__ == "__main__":
    main()