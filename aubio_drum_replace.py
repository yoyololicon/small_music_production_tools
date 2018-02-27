import argparse
from aubio import source, onset
from fractions import gcd
import numpy as np
import soundfile as sf
import pretty_midi
from scipy.signal import resample_poly

parser = argparse.ArgumentParser(description='Replace drum hits in the input file with samples.')
parser.add_argument('infile')
parser.add_argument('outfile')
parser.add_argument('--samples', nargs='+', help='Your own sample files. If no files, the default output format is midi')
parser.add_argument('-t', type=float, help='onset detection threshold. Default is -30db')
parser.add_argument('--tempo', type=float, help='The tempo you want to write into midi. Default is 120.')

if __name__ == '__main__':
    args = parser.parse_args()
    win_s = 1024
    hop_s = 256
    round_robin = 0
    bpm = 120.
    samples = []

    if args.samples:
        round_robin = len(args.samples)
    if args.tempo:
        bpm = args.tempo
    
    s = source(args.infile, 0, hop_s)
    samplerate = s.samplerate
    channels = s.channels
    
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

    if round_robin > 0: ##output wav
        for f in args.samples:
            data, sr = sf.read(f)
            if len(data.shape) > 1:
                channels = max(channels, data.shape[1])
            else:
                data = data[:, np.newaxis]

            if sr != samplerate:
                common = gcd(sr, samplerate)
                data = resample_poly(data, samplerate / common, sr / common, axis=0)

            samples.append(data)

        interval = [onsets[i] - onsets[i-1] for i in range(1, len(onsets))]
        total_length = onsets[-1] + samples[(len(onsets) % round_robin-1+round_robin)%round_robin].shape[0]

        f = np.zeros([total_length, channels])

        index = 0
        for ost in onsets:
            f[ost:ost+samples[index].shape[0], :] += samples[index]
            index = (index + 1) % round_robin

        print ('output file %s have %d channels' % (args.outfile, channels))
        norm_factor = np.max(f)
        if norm_factor > 1:
            f /= norm_factor
        sf.write(args.outfile, np.squeeze(f), samplerate)
    else:   ##output midi
        file = pretty_midi.PrettyMIDI(resolution=960, initial_tempo=bpm)
        drum_prog = pretty_midi.instrument_name_to_program('Steel Drums')
        trig_drum = pretty_midi.Instrument(program=drum_prog)

        for ost in onsets:
            time = float(ost)/samplerate
            note = pretty_midi.Note(velocity=127, pitch=36, start=time, end=time+0.001)
            trig_drum.notes.append(note)

        file.instruments.append(trig_drum)
        file.write(args.outfile)