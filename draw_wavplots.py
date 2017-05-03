import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import math
import numpy as np
import scipy.io.wavfile
from argparse import ArgumentParser

def plot_wave(x, title, xlabel, ylabel, filename):
    plt.plot(x)
    plt.suptitle(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(filename, format=filename.split('.')[-1])
    plt.show()

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-i1', '--infile1', help="Input wave for microphone1") 
    parser.add_argument('-i2', '--infile2', help="Input wave for microphone2") 
    parser.add_argument('-i3', '--infile3', help="Input wave for microphone3") 
    args = parser.parse_args()

    sampling_rate1, data1 = scipy.io.wavfile.read(args.infile1)
    sampling_rate2, data2 = scipy.io.wavfile.read(args.infile2)
    sampling_rate3, data3 = scipy.io.wavfile.read(args.infile3)

    plot_wave(data1[100:200], "Wave recorded in microphone 1", "time step", "frequency", "plots/microphone1.png")
    plot_wave(data2[100:200], "Wave recorded in microphone 2", "time step", "frequency", "plots/microphone2.png")
    plot_wave(data3[100:200], "Wave recorded in microphone 3", "time step", "frequency", "plots/microphone3.png")
    
