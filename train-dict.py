import numpy as np
import scipy.signal as signal
import os.path
from pickle import dump
from scipy.io import wavfile
from argparse import ArgumentParser
from sklearn.decomposition import NMF
from sklearn.preprocessing import minmax_scale

if __name__ == '__main__':
    # Config
    num_of_basis = 5
    stft_segment = 128

    # Read arguments
    parser = ArgumentParser()
    parser.add_argument('dict', help='dictionary output')
    parser.add_argument('input', nargs='+', help='audio file input(s)')
    args = parser.parse_args()

    long_spectrogram = np.zeros((stft_segment//2+1, (16000//(stft_segment//2)+1)*len(args.input)))

    model = NMF(n_components=num_of_basis)

    for idx, file in enumerate(args.input):
        rate, sample = wavfile.read(file)

        f, t, Zxx = signal.stft(sample, fs=rate, nperseg=stft_segment, window='hamming')
        power_spectrogram = minmax_scale(np.power(np.absolute(Zxx), 2), axis=1)
        long_spectrogram[:, idx*251:idx*251+251] = power_spectrogram

    model.fit(long_spectrogram.T)

    # Save dictionary
    with open(args.dict, 'wb') as f:
        dump(model.components_, f)
