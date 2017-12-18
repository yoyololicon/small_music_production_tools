import argparse
from fractions import gcd
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly, fftconvolve

parser = argparse.ArgumentParser(description='Impulse response loader.')
parser.add_argument('infile')
parser.add_argument('outfile')
parser.add_argument('IR')

if __name__ == '__main__':
    args = parser.parse_args()
    
    infile, infile_sr = sf.read(args.infile)
    ir, ir_sr = sf.read(args.IR)
    
    if ir_sr != infile_sr:
        common = gcd(infile_sr, ir_sr)
        ir = resample_poly(ir, infile_sr/common, ir_sr/common, axis=0)
    
    if len(infile.shape) > 1:
        channels = infile.shape[1]
    else:
        infile = infile[:, np.newaxis]
        channels = 1
        
    if len(ir.shape) > 1:
        channels = max(ir.shape[1], channels)
    else:
        ir = ir[:, np.newaxis]
        
    if infile.shape[1] > ir.shape[1]:
        ir = np.hstack((ir, ir))
    elif ir.shape[1] > infile.shape[1]:
        infile = np.hstack((infile, infile))
    
    s = len(ir)
    insize = len(infile)
    outsize = insize+s-1
    outfile = np.zeros([outsize, channels])
    
    for i in range(insize/s+1):
        for j in range(infile.shape[1]):
            outfile[i*s:(i+2)*s-1, j] += fftconvolve(infile[i*s:(i+1)*s, j], ir[:, j], mode='full')
    
    max = np.max(np.abs(outfile))
    outfile /= max
    sf.write(args.outfile, np.squeeze(outfile), infile_sr)
