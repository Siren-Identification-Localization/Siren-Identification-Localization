import numpy as np
import scipy.signal as signal
import os.path
from pickle import load, dump
from scipy.io import wavfile
from argparse import ArgumentParser
from sklearn.preprocessing import minmax_scale
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

if __name__ == '__main__':
    # Config
    stft_segment = 128

    # Read arguments
    parser = ArgumentParser()
    parser.add_argument('-d', '--dimred', default=None, help='dimensionality reduction model')
    parser.add_argument('output', help='classifier output')
    parser.add_argument('input', nargs='+', help='audio file input(s)')
    args = parser.parse_args()

    # Load dimensionality reduction model
    dimred = None
    if args.dimred is not None:
        with open(args.dimred, 'rb') as f:
            dimred = load(f)

    X = []
    y = []
    for idx, file in enumerate(args.input):
        rate, sample = wavfile.read(file)

        class_name = os.path.basename(os.path.dirname(file))
        if class_name == 'ambulance':
            y.append(1)
        else:
            y.append(0)

        f, t, Zxx = signal.stft(sample, fs=rate, nperseg=stft_segment, window='hamming')
        power_spectrogram = minmax_scale(np.power(np.absolute(Zxx), 2), axis=1)

        if dimred is not None:
            W = dimred.transform(power_spectrogram.T)
            X.append(np.ravel(W))
        else:
            X.append(np.ravel(power_spectrogram))

    classifier = GaussianNB()
    y_hat = classifier.fit(X, y).predict(X)
    print('Training accuracy: {}'.format(accuracy_score(y, y_hat)))

    # Save classifier
    with open(args.output, 'wb') as f:
        dump([dimred, classifier], f)
