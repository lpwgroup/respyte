import os
import subprocess
import numpy as np
from molecule_resp import Molecule_HJ
from collections import OrderedDict
from warnings import warn
AtomicMass = OrderedDict([('H' , 1.0079), ('He' , 4.0026),
                          ('Li' , 6.941), ('Be' , 9.0122), ('B' , 10.811), ('C' , 12.0107), ('N' , 14.0067), ('O' , 15.9994), ('F' , 18.9984), ('Ne' , 20.1797),
                          ('Na' , 22.9897), ('Mg' , 24.305), ('Al' , 26.9815), ('Si' , 28.0855), ('P' , 30.9738), ('S' , 32.065), ('Cl' , 35.453), ('Ar' , 39.948),
                          ('K' , 39.0983), ('Ca' , 40.078), ('Sc' , 44.9559), ('Ti' , 47.867), ('V' , 50.9415), ('Cr' , 51.9961), ('Mn' , 54.938), ('Fe' , 55.845), ('Co' , 58.9332),
                          ('Ni' , 58.6934), ('Cu' , 63.546), ('Zn' , 65.39), ('Ga' , 69.723), ('Ge' , 72.64), ('As' , 74.9216), ('Se' , 78.96), ('Br' , 79.904), ('Kr' , 83.8),
                          ('Rb' , 85.4678), ('Sr' , 87.62), ('Y' , 88.9059), ('Zr' , 91.224), ('Nb' , 92.9064), ('Mo' , 95.94), ('Tc' , 98), ('Ru' , 101.07), ('Rh' , 102.9055),
                          ('Pd' , 106.42), ('Ag' , 107.8682), ('Cd' , 112.411), ('In' , 114.818), ('Sn' , 118.71), ('Sb' , 121.76), ('Te' , 127.6), ('I' , 126.9045), ('Xe' , 131.293),
                          ('Cs' , 132.9055), ('Ba' , 137.327), ('La' , 138.9055), ('Ce' , 140.116), ('Pr' , 140.9077), ('Nd' , 144.24), ('Pm' , 145), ('Sm' , 150.36),
                          ('Eu' , 151.964), ('Gd' , 157.25), ('Tb' , 158.9253), ('Dy' , 162.5), ('Ho' , 164.9303), ('Er' , 167.259), ('Tm' , 168.9342), ('Yb' , 173.04),
                          ('Lu' , 174.967), ('Hf' , 178.49), ('Ta' , 180.9479), ('W' , 183.84), ('Re' , 186.207), ('Os' , 190.23), ('Ir' , 192.217), ('Pt' , 195.078),
                          ('Au' , 196.9665), ('Hg' , 200.59), ('Tl' , 204.3833), ('Pb' , 207.2), ('Bi' , 208.9804), ('Po' , 209), ('At' , 210), ('Rn' , 222),
                          ('Fr' , 223), ('Ra' , 226), ('Ac' , 227), ('Th' , 232.0381), ('Pa' , 231.0359), ('U' , 238.0289), ('Np' , 237), ('Pu' , 244),
                          ('Am' , 243), ('Cm' , 247), ('Bk' , 247), ('Cf' , 251), ('Es' , 252), ('Fm' , 257), ('Md' , 258), ('No' , 259),
                          ('Lr' , 262), ('Rf' , 261), ('Db' , 262), ('Sg' , 266), ('Bh' , 264), ('Hs' , 277), ('Mt' , 268)])

class Engine(object):
    def __init__(self, coordfile = None, input_file = None):
        if input_file is not None:
            self.read_input(input_file)

        if coordfile is not None:
            self.write_input(coordfile)

    def read_input(self, input_file):
        raise NotImplementedError

    def write_input(self, coordfile):
        raise NotImplementedError

    def run(self, cmd, input_files=None, output_files = None, job_path = None):
        if input_files is None or output_files is None:
            print('run nothing')
        if job_path is not None:
            os.chdir(job_path)

        subprocess.run([cmd, input_files, output_files], shell = True)

psi4_template_head = """memory 12 gb
molecule meoh {{
noreorient
nocom
{chg} {mult}
"""
psi4_template_tail="""}
set {
basis 6-31g*
reference uhf
}
E,wfn = prop('HF' , properties = ['GRID_ESP', 'GRID_FIELD'], return_wfn=True)
"""
bohr2Ang = 0.52918825

class EnginePsi4(Engine):
    def __init__(self, coordfile = None, input_file = None):
        if input_file is not None:
            self.read_input(input_file)

        if coordfile is not None:
            self.write_input(coordfile)

    def read_input(self, input_file):
        raise NotImplementedError

    def write_input(self,  coordfile, charge = 0, filename = 'input.dat', job_path = None):
        # take coordinate file and make molecule object ??
        if job_path is not None:
            os.chdir(job_path)

        molecule = Molecule_HJ()
        if coordfile.endswith('.xyz'):
            molecule.addXyzFile(coordfile)
        elif coordfile.endswith('.pdb'):
            molecule.addPdbFiles(coordfile)

        atomnum = molecule.elems[0] # should have molecule object for single molecule
        elem = []
        for i in atomnum:
            atomname = list(AtomicMass.keys())[i-1]
            elem.append(atomname)
        xyzs = molecule.xyzs[0]*bohr2Ang
        # totchg = 0
        # for indices, residue, charge in molecule.listofchargeinfo[0]:
        #     totchg += charge
        totchg = charge
        if totchg % 2 == 0:
            spinmult = 1
        elif totchg % 2 == 1:
            spinmult = 2
        def format_xyz_coord(element,xyz):
            return "%-s % 10.4f % 10.4f % 10.4f" % (element, xyz[0], xyz[1],xyz[2])

        with open(filename, 'w') as outfile:
            outfile.write(psi4_template_head.format(chg = totchg, mult = spinmult))
            for idx, i in enumerate(elem):
                outfile.write(format_xyz_coord(i, xyzs[idx]))
                outfile.write('\n')
            outfile.write(psi4_template_tail)
    def genespf(self, gridfile, espfile, effile, outputfile):
        ngrid = 0
        gridxyz = []
        espvals = []
        efvals = []
        with open(gridfile) as gf:
            for line in gf:
                ls = line.strip().expandtabs()
                sline = ls.split()
                gridxyz.append([float(i) for i in sline[0:3]])
        with open(espfile) as espf:
            for line in espf:
                ls = line.strip().expandtabs()
                sline = ls.split()
                espvals.append(float(sline[0]))
        with open(effile) as eff:
            for line in eff:
                ls = line.strip().expandtabs()
                sline = ls.split()
                efvals.append([float(i) for i in sline[0:3]])
        with open(outputfile, 'w') as outf:
            for i in range(len(gridxyz)):
                outf.write("% 15.10f % 15.10f % 15.10f % 15.10f \n" % (gridxyz[i][0], gridxyz[i][1], gridxyz[i][2], espvals[i]))
                outf.write("% 15.10f % 15.10f % 15.10f \n" % (efvals[i][0], efvals[i][1], efvals[i][2]))

    def espcal(self,  job_path):
        """ run esp calculation"""
        self.run('psi4 input.dat -o output.dat', input_files='input.dat', output_files='output.dat', job_path=job_path)

# def main():
