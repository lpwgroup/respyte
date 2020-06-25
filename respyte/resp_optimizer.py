from respyte.molecule import *
from respyte.readinp_resp import Input
from respyte.molecule_resp import *
from respyte.select_grid import *
from respyte.resp_unit import *

def main():
    print('\n\033[1m#======================================================================#')
    print('#|                         Welcome to respyte,                        |#')
    print('#|               a python implementation of RESP method               |#')
    print('#======================================================================#\033[0m')

    import argparse, sys
    parser = argparse.ArgumentParser(description="ESP based atomic partial charge generator for MM simulation", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('inputfolder', type=str, help='Input folder with specific directory structure')
    args = parser.parse_args()
    # print input command for reproducibility
    print( 'Command: ' +' '.join(sys.argv))

    cwd = os.getcwd()

    input_file = os.path.join(args.inputfolder, 'respyte.yml')
    molecules_dir = os.path.join(args.inputfolder, 'molecules')
    assert os.path.isdir(args.inputfolder), f'{args.inputfolder} not exist' 
    assert os.path.isdir(molecules_dir), f'{molecules_dir} not exist'
    assert os.path.isfile(input_file), f'{input_file} not exist'
    # read respyte.yml
    inp = Input(input_file)

    # Create molecule object
    if inp.cheminformatics == 'openeye':
        molecule = Molecule_OEMol()
    elif inp.cheminformatics == 'rdkit':
        molecule = Molecule_RDMol()
    else:
        print(' Not using Cheminformatics?')
        molecule = Molecule_respyte()

    molecule.addInp(inp)
    
    for idx, i in enumerate(inp.nmols):
        molN = 'mol%d' % (idx+1)
        wkd = os.path.join(molecules_dir, molN )
        coordfilepath = []
        espffilepath = []
        listofselectedPts = []
        for j in range(i):
            confN = 'conf%d' % (j+1)
            path = os.path.join(wkd, confN)
            pdbfile = path + '/%s_%s.pdb' % (molN, confN)
            mol2file = path + '/%s_%s.mol2' % (molN, confN)
            xyzfile = path + '/%s_%s.xyz' % (molN, confN)
            if os.path.isfile(pdbfile):
                coordpath = pdbfile
                coordfilepath.append(coordpath)
            elif os.path.isfile(mol2file):
                coordpath = mol2file
                coordfilepath.append(coordpath)
            elif os.path.isfile(xyzfile):
                coordpath = xyzfile
                coordfilepath.append(coordpath)
                # print(' This folder doesn not contain pdb or mol2 file format. ')
            else:
                raise RuntimeError(" Coordinate file should have pdb or mol2 file format! ")

            espfpath = path + '/%s_%s.espf' %(molN, confN)
            if not os.path.isfile(espfpath):
                raise RuntimeError(' %s file doesnt exist!!! '% espfpath)
            else:
                espffilepath.append(espfpath)

            if 'boundary_select' in inp.gridinfo:
                radii = inp.gridinfo['boundary_select']['radii']
                inner = inp.gridinfo['boundary_select']['inner']
                outer = inp.gridinfo['boundary_select']['outer']
                tmpmol = Molecule(coordpath)
                pts = []
                with open(espfpath, 'r') as espff:
                    for i, line in enumerate(espff):
                        fields = line.strip().split()
                        numbers = [float(field) for field in fields]
                        if (len(numbers)==4):
                            xyz = [x for x in numbers[0:3]]
                            pts.append(xyz)
                selectedPtsIdx, selectedPts = SelectGridPts(tmpmol,inner,outer,pts, radii )
            else:
                selectedPtsIdx = None
            listofselectedPts.append(selectedPtsIdx)
        molecule.addCoordFiles(*coordfilepath)
        molecule.addEspf(*espffilepath, selectedPts = listofselectedPts)

    os.chdir(cwd)
    cal = Respyte_Optimizer()
    cal.addInp(inp)
    cal.addMolecule(molecule)
    cal.run() 

if __name__ == '__main__':
    main()
