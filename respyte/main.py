from respyte.parse import *
from respyte.molecules import *
from respyte.objective import *
from respyte.optimizer import *
from respyte.procedure import *

def main():
    print('\n\033[1m#======================================================================#')
    print('#|                         Welcome to respyte,                        |#')
    print('#|               a python implementation of RESP method               |#')
    print('#======================================================================#\033[0m')

    import argparse, sys
    parser = argparse.ArgumentParser(description="ESP based atomic partial charge parameter generator for MM simulation", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('inputfolder', type=str, help='Input folder with specific directory structure')
    args = parser.parse_args()

    # print input command for reproducibility
    print( 'Command: ' +' '.join(sys.argv))

    cwd = os.getcwd()

    input_file = os.path.join(args.inputfolder, 'respyte.yml')
    molecules_dir = os.path.join(args.inputfolder, 'molecules')
    assert os.path.isdir(args.inputfolder), f'input folder, {args.inputfolder} not exist!'
    assert os.path.isdir(molecules_dir), f'molecule directory, {molecules_dir} not exist!'
    assert os.path.isfile(input_file), f'input file, {input_file} not exist!'

    # read respyte.yml 
    inp = Input(input_file)

    # generate molecules  object from input
    molecules = respyte_molecules()
    molecules.from_input(inp)
    # print(f'penalty: {inp.penalty}')
    resp(molecules, inp.symmetry, inp.model, inp.penalty, inp.procedure,  output_path = cwd)

if __name__ == '__main__':
    main()
