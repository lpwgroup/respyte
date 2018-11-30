# ForceBalance is a dependency
from molecule import *
from readinp import Input
from molecule_resp import Molecule_HJ, Molecule_OEMol
from select_grid import *

from resp_unit import *

def main():

    inp = Input()
    inp.readinput('input/respyte.yml')
    # inner, outer boundary

    if inp.cheminformatics =='openeye':
        molecule = Molecule_OEMol()
    else:
        molecule = Molecule_HJ()
    molecule.addInp(inp)

    cwd = os.getcwd() # the path where resp_optimizer.py is sitting

    for idx, i in enumerate(inp.nmols): # need to trim a bit more:
        molN = 'mol%d' % (idx+1)
        wkd = '%s/input/molecules/%s/' %(cwd,molN)
        coordfilepath = []
        espffilepath = []
        listofselectedPts = []
        for j in range(i):
            confN = 'conf%d' % (j+1)
            path = wkd + '%s/' % (confN)

            if os.path.isfile( path + '%s_%s.pdb' % (molN, confN)):
                print('Found %s_%s.pdb' % (molN, confN) )
                coordpath = path + '%s_%s.pdb' % (molN, confN)
                ftype = 'pdb'
                coordfilepath.append(coordpath)
            elif os.path.isfile( path + '%s_%s.xyz' % (molN, confN)):
                print('Found %s_%s.xyz' % (molN, confN) )
                coordpath = path + '%s_%s.xyz' % (molN, confN)
                ftype = 'xyz'
                coordfilepath.append(coordpath)

            espfpath = path + '%s_%s.espf' %(molN, confN)
            if not os.path.isfile(espfpath):
                raise RuntimeError('%s file doesnt exist!!! '% espfpath)
            else:
                espffilepath.append(espfpath)
            # listofselectedPts
            if 'boundary_select' in inp.gridinfo:
                inner = inp.gridinfo['boundary_select']['inner']
                outer = inp.gridinfo['boundary_select']['outer']
                radii = inp.gridinfo['radii']
                # if inp.gridinfo['radii'] =='bondi':
                #     radii = 'BondiRadii'
                # elif inp.gridinfo['radii'] == 'Alvarez':
                #     radii = 'AlvarezRadii'
                tmpmol = Molecule(coordpath)
                pts = []
                with open(espfpath,'r') as espffs:
                    for i, line in enumerate(espffs):
                        fields = line.strip().split()
                        numbers = [float(field) for field in fields]
                        if (len(numbers)==4):
                            xyz = [x for x in numbers[0:3]]
                            pts.append(xyz)
                selectedPts = SelectGridPts(tmpmol,inner,outer,pts, radii )
                print('Read %d pts from %s_%s.espf' % (len(pts), molN, confN))
                print('select pts: inner =%0.4f, outer = %0.4f' %(inner, outer))
                print('Use %d pts out of %d pts' %(len(selectedPts), len(pts)))
            else:
                selectedPts = None
            listofselectedPts.append(selectedPts)
        if ftype is 'xyz':
            # print(coordfilepath)
            molecule.addXyzFiles(*coordfilepath)
        elif ftype is 'pdb':
            molecule.addPdbFiles(*coordfilepath)
        # print('check')
        # print([len(listofselectedPts[i]) for i in range(5)]); input()
        # print(listofselectedPts)
        # input()
        molecule.addEspf(*espffilepath, selectedPts = listofselectedPts) # changed



    os.chdir(cwd) ####Let see if it's right...
    print('resp calculation is running on %s' % cwd)
    cal = Respyte_Optimizer()
    cal.addInp(inp)
    cal.addMolecule(molecule)
    if inp.restraintinfo:
        if inp.restraintinfo['penalty'] == 'model2':
            aval = inp.restraintinfo['a']
            cal.Model2qpot(aval)
        elif inp.restraintinfo['penalty'] == 'model3':
            aval = inp.restraintinfo['a']
            bval = inp.restraintinfo['b']
            cal.Model3qpot(aval, bval)
        elif inp.restraintinfo['penalty'] == '2-stg-fit':
            a1val = inp.restraintinfo['a1']
            a2val = inp.restraintinfo['a2']
            bval = inp.restraintinfo['b']
            cal.twoStageFit(a1val, a2val,bval)

    # print()
    # print('    #####################################################')
    # print('    ###               Test calculations               ###')
    # print('    #####################################################')
    # cal.Model2qpot(0.005)
    # cal.Model3qpot(0.0005,0.1)
    # cal.twoStageFit(0.0005,0.001,0.1)

if __name__ == '__main__':
    main()
