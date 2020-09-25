import yaml
from warnings import warn
import numpy as np

class Input:
    def __init__(self, inputFile=None):
        if inputFile is None:
            self.cheminformatics = None
            self.nmols = []
            self.charges = []
            self.settings = {}
            self.gridOptions = {}
        else:
            self.readinp(inputFile)
            self.gridOptions = self.genGridOptions(self.settings)

    def readinp(self, inputFile):
        inp = yaml.load(open(inputFile))
        # Read cheminformatics
        if 'cheminformatics' in inp:
            if inp['cheminformatics'] == 'openeye' or inp['cheminformatics'] == 'rdkit' or inp['cheminformatics'] == 'None':
                cheminformatics = inp['cheminformatics']
            else:
                raise NotImplementedError("%s is not implemented. Please choose openeye, rdkit or None:) " % inp['cheminformatics'])
        else:
            cheminformatics = None
        # Check how many molecules and how many conformers are provided.
        nmols = []
        names = []
        for mol in inp['molecules']:
            names.append(mol) # store mol names to match with charge information
            nmols.append(int(inp['molecules'][mol]))
        charges = np.zeros((len(names)))
        if 'charges' in inp:
            for mol in inp['charges']:
                idx = names.index(mol)
                charges[idx] = float(inp['charges'][mol])
        else:
            warn(' You didnt provide charges of each species. By default, all molecules are set to be neutral.')

        if 'grid_setting' in inp:
            settings = inp['grid_setting']
        else:
            # If grid_setting is not provided, use default setting.
            settings = {'type'   : 'msk',
                        'radii'  : 'bondi',
                        'method' : 'hf',
                        'basis'  : '6-31g*',
                        'pcm'    : 'N'}

        self.cheminformatics = cheminformatics
        self.nmols           = nmols
        self.charges          = charges
        self.settings        = settings

    def genGridOptions(self, settings):
        gridOptions = {}
        gridOptions['space'] = 0.7
        gridOptions['inner'] = 1.4
        gridOptions['outer'] = 2.0
        gridType = settings['type']
        if gridType == 'extendedmsk' or gridType == 'extendedMsk':
            gridOptions['gridType'] = 'extendedMsk'
            gridOptions['space'] = 1.0
            gridOptions['inner'] = 0.8
            gridOptions['outer'] = 2.4 # changed
        elif gridType=='MSK' or gridType=='msk':
            gridOptions['gridType'] = 'MSK'
            gridOptions['space'] = 1.0
        elif gridType == 'newfcc':
            gridOptions['gridType'] = 'shellFac'
            gridOptions['inner'] = 0.8
            gridOptions['outer'] = 2.4 # changed
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

        if 'inner' in settings:
            gridOptions['inner'] = settings['inner']
        if 'outer' in settings:
            gridOptions['outer'] = settings['outer']
        if 'space' in settings:
            gridOptions['space'] = settings['space']
        return gridOptions

def main():
    inp = Input(sys.argv[1])
    print(inp.gridOptions)
if __name__ == "__main__":
    main()
