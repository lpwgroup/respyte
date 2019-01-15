"""
resp_optimizer.py
Implementation of open-source vrsion of RESP method

"""
import os, sys
import numpy as np
from shutil import copyfile
from readinp_esp import *
from readmol import * # read molecules and assign (what?)
from molecule import * # copied from forcebalance package and modified
from gen_grid_pts import * # take either rdkit ol or oechem mol to generate grid points around molecules (modified CIB's code)
from engine import *

def main():
    # cwd = current working directory in which input folder exists
    cwd = os.getcwd()
    # check if the current working idr contains input folder
    print('\n')
    print('---------------------------------------------------------')
    print(' 1. Reading  input files and folders.                    ')
    print('---------------------------------------------------------')
    if os.path.isdir("%s/input" % cwd):
        print(' Found the input folder. Now read input/input.yml')
    else:
        print(' Failed to find input folder. Should have input(folder) containing input.yml and molecules(folder)')
    # read input.yml and generate gridOption.
    inp = Input('%s/input/input.yml' % (cwd))
    gridOptions = inp.gridOptions

    # Visit all subfolder molecule/ moli/confj and generate grid.dat
    for idx, i in enumerate(inp.nmols):
        molN = 'mol%d' % (idx+1)
        wkd = '%s/input/molecules/%s' % (cwd, molN)
        for j in range(i):
            confN = 'conf%d' % (j+1)
            path = wkd + '/%s' % (confN)
            # gen molecule object from pdb file
            pdbfile = path + '/%s_%s.pdb' % (molN, confN)
            mol2file = path + '/%s_%s.mol2' % (molN, confN)
            xyzfile = path + '/%s_%s.xyz' % (molN, confN)
            if os.path.isfile(pdbfile):
                coordfile = pdbfile
                #mol = Molecule(pdbfile)
            elif os.path.isfile(mol2file):
                coordfile = mol2file
                #mol = Molecule(mol2file)
            elif os.path.isfile(xyzfile):
                print(" Warning: Are you using xyz files? xyz format is not recommendable.")
                print("          But no worries! It will convert your xyz files to pdb with single residue named 'MOL'.")
                xyztopdb(xyzfile)
                coordfile = pdbfile
            else:
                raise RuntimeError(" Coordinate file does not exist! ")
            # Assign radii
            radii = inp.settings['radii']
            if inp.cheminformatics == 'openeye':
                mol = ReadOEMolFromFile(coordfile)
                mol = assignRadii(mol, 'OEMol', radii)
                moltype = 'OEMol'
            elif inp.cheminformatics == 'rdkit':
                mol = ReadRdMolFromFile(coordfile)
                mol = assignRadii(mol, 'RDMol', radii)
                moltype = 'RDMol'
            else:
                mol = ReadMolFromFile(coordfile)
                mol = assignRadii(mol, 'FBMol', radii)
                moltype = 'FBMol'
            # Generate points
            print('\n')
            print('---------------------------------------------------------')
            print(' 2. Generating grid points around each molecules.        ')
            print('---------------------------------------------------------')

            if moltype is 'FBMol':
                raise NotImplementedError('Grid gen using fbmol not implemented yet! Sorry!')
            if gridOptions['gridType'] == 'MSK' or gridOptions['gridType']=='extendedMsk':
                # generate Mers-Singh-Kollman Connolly surfaces at 1.4, 1.6, 1.8 and 2.0 * vdW radii
                allPts = GenerateMSKShellPts(mol, gridOptions, moltype)
            else:
                # Generate a face-centered cubic grid shell around the molecule using gridOptions
                allPts = [GenerateGridPointSetAroundOEMol(mol, gridOptions, moltype)]
            print('Total points:', np.sum([len(i) for i in allPts]))

            # Write grid.dat
            print('\n')
            print('---------------------------------------------------------')
            print(' Psi calculations are done. Will create espf files.      ')
            print('---------------------------------------------------------')
            tmppath = '%s/tmp' % path
            if inp.settings['forcegen'] == 'N' and os.path.isfile('%s/grid.dat' % path):
                if os.path.isdir(tmppath):
                    print('Rename pre-existing tmp dir into tmp.0 and re-make tmp dir.')
                    os.rename(tmppath, '%s/tmp.0' % path)
                    os.mkdir(tmppath)
                    copyfile('%s/grid.dat' % path, '%s/grid.dat' % tmppath)
                else:
                    os.mkdir(tmppath)
                    # copy grid.dat into tmp dir
                    copyfile('%s/grid.dat' % path, '%s/grid.dat' % tmppath)
                print(' Pre-existing grid.dat has been copied into tmp for Psi4 calculation.')
            else:
                if os.path.isdir(tmppath):
                    print(' Rename pre-existing tmp dir into tmp.0 and re-make tmp dir.')
                    os.rename(tmppath, '%s/tmp.0' % path)
                os.mkdir(tmppath)
                ofs = open('%s/grid.dat' % tmppath,'w')

                for pts in allPts: #changed (for msk-based scheme, allPts is a list of lists)
                    for pt in pts:
                        ofs.write( '{0:10.6f} {1:10.6f} {2:10.6f}\n'.format(pt[0], pt[1], pt[2]) )
                ofs.close()
                print(' grid.dat is successfully generated in %s.' % tmppath)
            # Run engine to run Psi4 calculation inside tmp directory.
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

            ### Need to add 'solvation' setting!
            if 'pcm' in inp.settings:
                if inp.settings['pcm'] is 'N':
                    solv = None
                elif inp.settings['pcm'] is 'Y':
                    solv = inp.settings['solvent']
            else:
                solv = None

            engine.write_input(coordfile, method = mthd, basis = bss, charge = chg, solvent=solv, job_path = tmppath)
            engine.espcal(job_path= tmppath)
            griddat = tmppath + '/grid.dat'
            espdat = tmppath + '/grid_esp.dat'
            efdat = tmppath + '/grid_field.dat'
            espfoutput = tmppath + '/%s_%s.espf' % (molN, confN)
            engine.genespf(griddat, espdat, efdat, espfoutput)
            # copy espf file generated in moli/confj/tmp to moli/confj/
            copyfile(espfoutput, path + '/%s_%s.espf' % (molN, confN))
            copyfile(griddat, path + '/grid.dat')
            print('\n')
            print('---------------------------------------------------------')
            print('       Done! Hope it helps, Will see you again:)         ')
            print('---------------------------------------------------------')

if __name__ == "__main__":
    main()
