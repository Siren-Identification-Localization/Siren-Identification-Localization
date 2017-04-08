import numpy as np
from scipy.io import wavfile
from argparse import ArgumentParser
from os import path

if __name__ == '__main__':
    # Read arguments
    parser = ArgumentParser()
    parser.add_argument('input', nargs='+', help='audio file input(s)')
    parser.add_argument('output', help='output directory')
    args = parser.parse_args()

    for file in args.input:
        filename, ext = path.splitext(path.basename(file))

        rate, sample = wavfile.read(file)

        for start in range(0, sample.size, rate//2):
            if start+rate < sample.size:
                chunk = sample[start:min(sample.size, start+rate)]
                wavfile.write('{}/{}_{}{}'.format(args.output, filename, start, ext), rate, chunk)
