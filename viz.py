import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
from scipy.io import wavfile
from argparse import ArgumentParser

if __name__ == '__main__':
    # Read arguments
    parser = ArgumentParser()
    parser.add_argument('input', help='audio file input')
    args = parser.parse_args()

    rate, sample = wavfile.read(args.input)

    f, t, Zxx = signal.stft(sample, fs=rate, nperseg=512, window='hamming')
    power_spectogram = np.power(np.absolute(Zxx), 2)

    plt.pcolormesh(t, f, power_spectogram)
    plt.show()
