import argparse
from aubio import source, onset
from fractions import gcd
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

parser = argparse.ArgumentParser(description='Replace drum hits in the input file with samples.')
parser.add_argument('infile')
parser.add_argument('outfile')
parser.add_argument('samples', nargs='+', help='your own sample files')
parser.add_argument('-t', type=float, help='onset detection threshold, default is -30db')

if __name__ == '__main__':
    args = parser.parse_args()
    win_s = 1024
    hop_s = 256

    round_robin = len(args.samples)
    samples = []
    
    s = source(args.infile, 0, hop_s)
    samplerate = s.samplerate
    channels = s.channels
    
    for f in args.samples:
        data, sr = sf.read(f)
        if len(data.shape) > 1:
            channels = max(channels, data.shape[1])
        else:
            data = data[:, np.newaxis]

        if sr != samplerate:
            common = gcd(sr, samplerate)
            data = resample_poly(data, samplerate/common, sr/common, axis=0)

        samples.append(data)
    
    o = onset("default", win_s, hop_s, samplerate)
    if not args.t:
        o.set_silence(-30)
    else:
        o.set_silence(args.t)
    o.set_threshold(0.4)
    # list of onsets, in samples
    onsets = []
    while True:
        sps, read = s()
        if o(sps):
            onsets.append(o.get_last())
        if read < hop_s: break
        
    print ('found %d onset position.' % len(onsets))
    
    interval = [onsets[i] - onsets[i-1] for i in range(1, len(onsets))]
    
    total_length = onsets[-1] + samples[(len(onsets) % round_robin-1+round_robin)%round_robin].shape[0]

    f = np.zeros([total_length, channels])

    index = 0
    for ost, i in zip(onsets[:-1], interval):
        if len(samples[index]) >= i:
            f[ost:ost+i, :] = samples[index][:i, :]
        else:
            f[ost:ost+samples[index].shape[0], :] = samples[index]
        index = (index + 1) % round_robin
    
    f[onsets[-1]:onsets[-1]+samples[index].shape[0], :] = samples[index]

    print ('output file %s have %d channels' % (args.outfile, channels))
    
    sf.write(args.outfile, np.squeeze(f), samplerate)
        