import numpy as np
import soundfile as sf
import argparse
from os import listdir
from os.path import join

parser = argparse.ArgumentParser(description='Combine splited stereo files in your ready to mix multitracks folder.'
                                             'The splited filename should end with L.wav or R.wav')
parser.add_argument('directory')
parser.add_argument('-outdir', help='Output folder location. Default is the same folder where your multitracks are.')

if __name__ == '__main__':
    args = parser.parse_args()

    matchL = [f[:-5] for f in listdir(args.directory) if f[-5:] == 'L.wav']
    matchR = [f[:-5] for f in listdir(args.directory) if f[-5:] == 'R.wav']
    match = list(set(matchL).intersection(matchR))
    if not args.outdir:
        output_dir = args.directory
    else:
        output_dir = args.outdir

    for f in match:
        lf = join(args.directory, f) + 'L.wav'
        rf = join(args.directory, f) + 'R.wav'

        left, sr = sf.read(lf)
        right, sr = sf.read(rf)
        sf.write(join(output_dir, f) + '.wav', np.hstack((left[:, np.newaxis], right[:, np.newaxis])), sr)
