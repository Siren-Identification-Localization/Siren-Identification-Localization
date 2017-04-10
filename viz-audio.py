import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
from os import path
from pickle import load
from scipy.io import wavfile
from argparse import ArgumentParser
from sklearn.preprocessing import minmax_scale
from sklearn.metrics import accuracy_score

if __name__ == '__main__':
    # Config
    stft_segment = 128

    # Read arguments
    parser = ArgumentParser()
    parser.add_argument('model', help='model file')
    parser.add_argument('input', nargs='+', help='audio file input(s)')
    parser.add_argument('--save', action='store_true', help='save plot instead of showing to the screen')
    args = parser.parse_args()

    # Load model
    with open(args.model, 'rb') as f:
        dimred, classifier = load(f)

    for file in args.input:
        filename, _ = path.splitext(path.basename(file))
        rate, sample = wavfile.read(file)

        detection = np.zeros((sample.size//rate+1, 2))
        for sec_idx, start in enumerate(range(0, sample.size, rate)):
            chunk = np.zeros(rate)
            chunk[0:min(sample.size-start, rate)] = sample[start:min(sample.size, start+rate)]

            f, t, Zxx = signal.stft(chunk, fs=rate, nperseg=stft_segment, window='hamming')
            power_spectrogram = minmax_scale(np.power(np.absolute(Zxx), 2), axis=1)

            if dimred is not None:
                W = dimred.transform(power_spectrogram.T)
                X = np.reshape(W, (1, -1))
            else:
                X = np.reshape(power_spectrogram, (1, -1))

            detection[sec_idx] = classifier.predict_proba(X)

        # Plot!
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, aspect=20)
        ax.set_title(file)
        major_ticks = np.arange(0, detection.shape[0], 10)
        minor_ticks = np.arange(0, detection.shape[0], 2)
        ax.set_xticks(major_ticks)
        ax.set_xticks(minor_ticks, minor=True)
        ax.grid(which='both')
        ax.grid(which='minor', alpha=0.2)
        ax.grid(which='major', alpha=0.5)
        ax.set_xlabel('Seconds')
        ax.pcolor(detection.T)
        if args.save:
            fig.savefig('{}_plot.png'.format(filename), dpi=200)
            plt.close()
        else:
            plt.show()
