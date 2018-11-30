import os, sys
import numpy as np
from shutil import copyfile
from readinp import *
from readmol import * # read molecule and assign
from molecule import * # copied from forcebalance package and modified
from gen_grid_pts import * # take either rdkit mol or oechem mol and gen grid points (modified CIB's code)
from engine import *
def main():

    cwd = os.getcwd()
    # check if the current working dir contains input folder
    if os.path.isdir("%s/input" % cwd):
        print('Found the input folder. Now read input/respyte.yml')
    else:
        print('Failed to find the input folder. Should have a folder named input containing respyte.yml and molecules.')
        raise RuntimeError

    # read input and generate gridOptions!
    inp = Input('%s/input/input.yml' % cwd)
    gridOptions = inp.gridOptions

    # visit all subfolder molecules/moli/confj and gen grid.dat
    for idx, i in enumerate(inp.nmols):
        molN = 'mol%d' % (idx+1)
        wkd = '%s/input/molecules/%s' % (cwd, molN)
        for j in range(i):
            confN = 'conf%d' % (j+1)
            path = wkd + '/%s' % (confN)

            # gen molecule object from pdb file
            pdbfile = path + '/%s_%s.pdb' % (molN, confN)
            xyzfile = path + '/%s_%s.xyz' % (molN, confN)
            if os.path.isfile(pdbfile):
                coordfile = pdbfile
                mol = Molecule(pdbfile)
            else: # generate pdb file from xyz file!
                raise NotImplementedError("cant generate grid.dat using xyz file yet:(")

            # Assign radii
            radii = inp.settings['radii']
            if inp.cheminformatics == 'openeye':
                mol = ReadOEMolFromFile(pdbfile)
                mol = assignRadii(mol, 'OEMol', radii)
                moltype = 'OEMol'
            elif inp.cheminformatics == 'rdkit':
                mol = ReadRdMolFromFile(pdbfile)
                mol = assignRadii(mol, 'RDMol', radii)
                moltype = 'RDMol'
            else:
                mol = ReadMolFromFile(pdbfile)
                mol = assignRadii(mol, 'FBMol', radii)
                moltype = 'FBMol'

            # generate points
            if moltype is 'FBMol':
                raise NotImplementedError('grid gen using fbmol not implemented yet! Sorry!')

            if gridOptions['gridType']=='MSK' or gridOptions['gridType'] =='extendedMsk':
                # generate Merz-Singh-Kollman Connolly surfaces at 1.4, 1.6, 1.8, and 2.0 * vdW radii
                allPts = GenerateMSKShellPts( mol, gridOptions, moltype)
            else:
                # generate a face-centered cubic grid shell around the molecule using gridOptions
                allPts = [GenerateGridPointSetAroundOEMol(mol, gridOptions, moltype)]
            print('Total points:', np.sum([len(allPts[i]) for i in range(len(allPts))]))

            # write grid.dat
            tmppath = '%s/tmp' % path
            if inp.settings['forcegen'] == 'N' and  os.path.isfile('%s/grid.dat' % path):
                pass
            else:
                os.mkdir(tmppath)
                ofs = open('%s/grid.dat' % tmppath,'w')
    
                for pts in allPts: #changed (for msk-based scheme, allPts is a list of lists)
                    for pt in pts:
                        ofs.write( '{0:10.6f} {1:10.6f} {2:10.6f}\n'.format(pt[0], pt[1], pt[2]) )
                ofs.close()
                print('grid.dat is generated in %s' % path)
                """
                    !!!ACHTUNG!!! Below this line is under construction !!!ACHTUNG!!!
                    !!!ACHTUNG!!! Below this line is under construction !!!ACHTUNG!!!
                    !!!ACHTUNG!!! Below this line is under construction !!!ACHTUNG!!!
                    !!!ACHTUNG!!! Below this line is under construction !!!ACHTUNG!!!
                """
                # run engine to run Psi4 calculation in tmp dir.
                engine = EnginePsi4()
                if 'method' in inp.settings:
                    mthd = inp.settings['method']
                else:
                    mthd = 'hf'
                if 'basis' in inp.settings:
                    bss = inp.settings['basis']
                else:
                    bss = '6-31g*'
    
                chg = inp.charges[idx]
    
                engine.write_input(coordfile, method = mthd, basis = bss, charge = chg, job_path = tmppath)
                engine.espcal(job_path= tmppath)
                griddat = tmppath + '/grid.dat'
                espdat = tmppath + '/grid_esp.dat'
                efdat = tmppath + '/grid_field.dat'
                espfoutput = tmppath + '/%s_%s.espf' % (molN, confN)
                engine.genespf(griddat, espdat, efdat, espfoutput)
                # copy espf file generated in moli/confj/tmp to moli/confj/
                copyfile(espfoutput, path + '/%s_%s.espf' % (molN, confN))
                copyfile(griddat, path + '/grid.dat')
                """
                    !!!ACHTUNG!!! Below this line is under construction !!!ACHTUNG!!!
                    !!!ACHTUNG!!! Below this line is under construction !!!ACHTUNG!!!
                    !!!ACHTUNG!!! Below this line is under construction !!!ACHTUNG!!!
                    !!!ACHTUNG!!! Below this line is under construction !!!ACHTUNG!!!
                """
if __name__ == '__main__':
    main()
