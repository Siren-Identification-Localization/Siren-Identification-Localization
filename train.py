import numpy as np
import scipy.signal as signal
from pickle import dump
from scipy.io import wavfile
from argparse import ArgumentParser
from sklearn.decomposition import NMF
from sklearn.preprocessing import minmax_scale

if __name__ == '__main__':
    # Config
    num_of_basis = 3
    stft_segment = 128

    # Read arguments
    parser = ArgumentParser()
    parser.add_argument('model', help='model output')
    parser.add_argument('input', nargs='+', help='audio file input(s)')
    args = parser.parse_args()

    Hs = []

    model = NMF(n_components=num_of_basis)

    for file in args.input:
        rate, sample = wavfile.read(file)

        f, t, Zxx = signal.stft(sample, fs=rate, nperseg=stft_segment, window='hamming')
        power_spectrogram = minmax_scale(np.power(np.absolute(Zxx), 2), axis=1)

        model.fit(power_spectrogram.T)
        Hs.append(model.components_)

    mean_H = np.mean(Hs, axis=0)

    # Save model
    with open(args.model, 'wb') as f:
        dump(mean_H, f)
