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
    T = np.array([[0.5, 0.5], [0.1, 0.9]])

    # Read arguments
    parser = ArgumentParser()
    parser.add_argument('model', help='model file')
    parser.add_argument('input', nargs='+', help='audio file input(s)')
    parser.add_argument('--save', action='store_true', help='save plot instead of showing to the screen')
    args = parser.parse_args()

    # Load model
    with open(args.model, 'rb') as f:
        dimred, classifier = load(f)

    previous_detection = 0
    for file in args.input:
        filename, _ = path.splitext(path.basename(file))
        rate, sample = wavfile.read(file)

        detection = np.zeros((sample.size//rate+1)*4)
        for sec_idx, start in enumerate(range(0, sample.size, rate//4)):
            chunk = np.zeros(rate)
            chunk[0:min(sample.size-start, rate)] = sample[start:min(sample.size, start+rate)]

            f, t, Zxx = signal.stft(chunk, fs=rate, nperseg=stft_segment, window='hamming')
            power_spectrogram = minmax_scale(np.power(np.absolute(Zxx), 2), axis=1)

            if dimred is not None:
                W = dimred.transform(power_spectrogram.T)
                X = np.reshape(W, (1, -1))
            else:
                X = np.reshape(power_spectrogram, (1, -1))

            probs = classifier.predict_proba(X)[0]
            smoothed_probs = probs * T[previous_detection]
            normalized_probs = smoothed_probs / np.sum(smoothed_probs)
            if sec_idx == 0:
                detection[(sec_idx):(sec_idx)+4] = normalized_probs[1]
            else:
                detection[(sec_idx)+3:(sec_idx)+4] = normalized_probs[1]
            previous_detection = np.argmax(normalized_probs)

        # Plot!
        fig = plt.figure()
        plt.suptitle(file)
        major_ticks = np.arange(0, detection.shape[0], 40)
        minor_ticks = np.arange(0, detection.shape[0], 4)

        ax = fig.add_subplot(1, 1, 1)
        ax.set_xticks(major_ticks)
        ax.set_xticks(minor_ticks, minor=True)
        ax.set_yticks([0, 1])
        ax.grid(which='both')
        ax.grid(which='minor', alpha=0.1)
        ax.grid(which='major', alpha=0.25)
        ax.set_xlabel('Seconds')
        ax.set_xticklabels([i for i in range(0, detection.shape[0], 10)])
        ax.set_ylim([-0.1, 1.1])
        ax.plot(detection)

        if args.save:
            fig.set_size_inches(5, 2)
            fig.savefig('{}_plot.png'.format(filename), dpi=200)
            plt.close()
        else:
            plt.show()
