import os, sys
import numpy as np
import shutil
from respyte.molecules import rdmolFromFile, assignRadiiRdmol
from respyte.fbmolecule import *
from respyte.parse_esp_gen_input import *
from respyte.gen_grid_pts import *
from respyte.engine import *

def generate_pts(coord_file, gridOptions):
    mol = rdmolFromFile(coord_file)
    mol = assignRadiiRdmol(mol, gridOptions['radii'])
    moltype = 'RDMol'        
    if gridOptions['gridType'] == 'MSK' or gridOptions['gridType']=='extendedMsk':
        # generate Mers-Singh-Kollman Connolly surfaces at 1.4, 1.6, 1.8 and 2.0 * vdW radii
        allPts = GenerateMSKShellPts(mol, gridOptions, moltype)
    else:
        # Generate a face-centered cubic grid shell around the molecule using gridOptions
        allPts = [GenerateGridPointSetAroundOEMol(mol, gridOptions, moltype)]     
    print(f'\033[1m  * Total points: {np.sum([len(i) for i in allPts])}\033[0m')
    return allPts 

def main():
    print('\033[1m#======================================================================#')
    print('#|                         Welcome to respyte,                        |#')
    print('#|               a python implementation of RESP method               |#')
    print('#======================================================================#\033[0m')

    import argparse, sys
    parser = argparse.ArgumentParser(description="ESPF file generation for charge fitting using PSI4", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('inputfolder', type=str, help='Input folder with specific directory structure')
    args = parser.parse_args()

    # print input command for reproducibility
    print(f"Command: {' '.join(sys.argv)}")

    print('---------------------------------------------------------')
    print(' 1. Reading  input files and folders.                    ')
    print('---------------------------------------------------------')

    input_file = os.path.join(args.inputfolder, 'input.yml')

    # 1. Read input.yml and generate input object
    inp = Input(input_file)

    # Visit all subfolder molecule/moli(molecule name)/confj and generate grid.dat for esp calculation
    for name, info in inp.mols.items():
        nconf = info['nconf']
        net_charge = info['net_charge']
        for j in range(1, nconf+1):
            confN = 'conf%d' % (j)
            path = os.path.join(inp.inp_dir, 'molecules', name, confN)
            molN_confN = '%s_%s' % (name, confN)
            pdbfile = os.path.join(path, '%s.pdb' % molN_confN)
            mol2file = os.path.join(path, '%s.mol2' % molN_confN)
            if os.path.isfile(pdbfile):
                coord_file = pdbfile
            elif os.path.isfile(mol2file):
                coord_file = mol2file
            else:
                raise RuntimeError(" Coordinate file should have pdb or mol2 file format! ")
               
            print('---------------------------------------------------------')
            print(' 2. Generating grid points around the input molecule.    ')
            print('---------------------------------------------------------')
            # generate grid points
            allPts = generate_pts(coord_file, inp.gridOptions)
  
            # Write the points into a grid.dat
            tmppath = os.path.join(path, 'tmp')
            if inp.settings['forcegen'] == 'N' and os.path.isfile(os.path.join(path, 'grid.dat')):
                # if already  have tmp file, rename it and gen new tmp for the new calc. 
                if os.path.isdir(tmppath):
                    # print(' Rename pre-existing tmp dir and re-make tmp dir for the new esp calculation')
                    tmp_folders = [f for f in os.listdir(path) if f.startswith('tmp_') and os.path.isdir(os.path.join(path,f))]
                    tmp_folders.sort()
                    last_tmp = tmp_folders[-1]  if len(tmp_folders) > 0 else 'tmp_0000'
                    next_idx = int(last_tmp[4:]) + 1
                    os.rename(tmppath, os.path.join(path,'tmp_%04i' % next_idx) )
                    os.mkdir(tmppath)
                    shutil.copyfile(os.path.join(path, 'grid.dat'), os.path.join(tmppath, 'grid.dat'))
                else:
                    os.mkdir(tmppath)
                    # copy grid.dat into tmp dir
                    shutil.copyfile(os.path.join(path, 'grid.dat'), os.path.join(tmppath, 'grid.dat'))
                print(' Pre-existing grid.dat will be used for Psi4 ESP calculation.')
            else:
                if os.path.isdir(tmppath):
                    # print('Rename pre-existing tmp dir and re-make tmp dir for the new esp calculation')
                    tmp_folders = [f for f in os.listdir(path) if f.startswith('tmp_') and os.path.isdir(os.path.join(path,f))]
                    tmp_folders.sort()
                    last_tmp = tmp_folders[-1]  if len(tmp_folders) > 0 else 'tmp_0000'
                    next_idx = int(last_tmp[4:]) + 1
                    os.rename(tmppath, os.path.join(path,'tmp_%04i' % next_idx) )
                os.mkdir(tmppath)
                ofs = open(os.path.join(tmppath, 'grid.dat'),'w')
                for pts in allPts: 
                    for pt in pts:
                        ofs.write( '{0:10.6f} {1:10.6f} {2:10.6f}\n'.format(pt[0], pt[1], pt[2]) )
                ofs.close()

            print('---------------------------------------------------------')
            print(' 3. Now running Psi ESP calculation.                     ')
            print('---------------------------------------------------------')
            # Run engine to run Psi4 calculation inside tmp directory.
            engine = EnginePsi4()
            if 'method' in inp.settings: ## maybe this will be moved to parsing script
                mthd = inp.settings['method']
            else:
                mthd = 'hf'
            if 'basis' in inp.settings:
                bss = inp.settings['basis']
            else:
                bss = '6-31g*'

            if 'pcm' in inp.settings:
                if inp.settings['pcm'] is 'N':
                    solv = None
                elif inp.settings['pcm'] is 'Y':
                    solv = inp.settings['solvent']
            else:
                solv = None

            engine.write_input(coord_file, method = mthd, basis = bss, charge = net_charge, solvent=solv, job_path = tmppath)
            engine.espcal(job_path= tmppath)
            griddat = os.path.join(tmppath,'grid.dat')
            espdat = os.path.join(tmppath,'grid_esp.dat')
            efdat = os.path.join(tmppath,'grid_field.dat')
            espfoutput = os.path.join(tmppath,'%s_%s.espf' % (name, confN))
            engine.genespf(griddat, espdat, efdat, espfoutput)
            # copy espf file generated in moli/confj/tmp to moli/confj/
            shutil.copyfile(espfoutput, os.path.join(path,'%s_%s.espf' % (name, confN)))
            shutil.copyfile(griddat, os.path.join(path,'grid.dat'))

    print('\n\033[1m Done! Hope it helps, Will see you later:)\033[0m')

if __name__ == "__main__":
    main()
