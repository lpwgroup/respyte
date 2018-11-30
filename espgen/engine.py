import os
import subprocess
import numpy as np
# from molecule_resp import Molecule_HJ
from collections import OrderedDict
from warnings import warn
from molecule import *

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
psi4_template_tail="""}}
set {{
basis {basis}
}}
E,wfn = prop('{method}' , properties = ['GRID_ESP', 'GRID_FIELD'], return_wfn=True)
"""

class EnginePsi4(Engine):
    def __init__(self, coordfile = None, input_file = None):
        if input_file is not None:
            self.read_input(input_file)

        if coordfile is not None:
            self.write_input(coordfile)

    def read_input(self, input_file):
        raise NotImplementedError

    def write_input(self, coordfile, basis='6-31g*', method = 'hf', charge = 0, filename = 'input.dat', job_path = None):
        # take coordinate file and make molecule object ??
        if job_path is not None:
            os.chdir(job_path)

        molecule =Molecule(coordfile)
        atomicnum =[list(PeriodicTable.keys()).index(molecule.elem[i])+1 for i in range(molecule.na)]
        # print(atomicnum)
        # input()
        totchg = int(charge)
        for el in atomicnum:
            totchg += int(el)
        if totchg % 2 == 0:
            spinmult = 1
        elif totchg % 2 == 1:
            spinmult = 2
        def format_xyz_coord(element,xyz):
            return "%-s % 10.4f % 10.4f % 10.4f" % (element, xyz[0], xyz[1],xyz[2])

        with open(filename, 'w') as outfile:
            outfile.write(psi4_template_head.format(chg = int(charge), mult = spinmult))
            for idx, i in enumerate(molecule.elem):
                outfile.write(format_xyz_coord(i, molecule.xyzs[0][idx]))
                outfile.write('\n')
            outfile.write(psi4_template_tail.format(basis = basis, method =method)) #### set inside input file


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
