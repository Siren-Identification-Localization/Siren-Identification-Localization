# name: Hankyu Jang
# student id: 2000108037

import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile
import sys
from argparse import ArgumentParser

N_frame = 1024
eps = np.spacing(1)

def plot_signal(x, sampling_rate, title, xlabel, ylabel, filename):
    plt.plot(x)
    plt.title(title)
    plt.xlabel("1/" + str(sampling_rate) + " seconds")
    plt.ylabel("frequency")
    plt.savefig(filename, format=filename.split('.')[-1])
    plt.show()

def create_DFT_matrix():
    F = np.ones((N_frame , N_frame ), dtype = complex)
    for f in range(N_frame ):
        for n in range(N_frame ):
            F[f, n] = np.exp(-1j * 2*np.pi*f*n / N_frame )
    return F

def hann_function(n, num_sample):
    return 0.5*(1-np.cos(2*np.pi*n / (num_sample-1)))

def prepare_data_matrix(data):
    window_size = N_frame /2
    num_col = int(np.ceil(len(data)/float(window_size)))
    data_with_eps = np.zeros(num_col*window_size)
    data_with_eps[0:len(data)] = data
    data_with_eps[len(data):] = eps

    # I will transpose the X matrix later
    X = np.full((num_col, N_frame ), eps)
    han_vector = np.ones(N_frame )

    for i in range(N_frame ):
        han_vector[i] = hann_function(i, N_frame )

    for i in range(num_col-1):
        segment = data_with_eps[i*window_size : i*window_size+N_frame ]
        X[i] = np.multiply(segment, han_vector)

    return X.T

def plot_matrix(X, title, xlabel, ylabel, filename):
    plt.matshow(X, cmap='plasma')
    plt.title(title)
    plt.xlabel('xlabel')
    plt.ylabel('ylabel')
    plt.savefig(filename, format=filename.split('.')[-1])
    plt.show()

def create_DFT_inverse():
    F_inv = np.ones((N_frame , N_frame ), dtype = complex)
    for f in range(N_frame ):
        for n in range(N_frame ):
            F_inv[f, n] = 1/float(N_frame ) * np.exp(1j * 2*np.pi*f*n / N_frame )
    return F_inv
     
def recover_time_domain_signal(X_recovered):
    window_size = N_frame /2
    num_col = X_recovered.shape[1]
    data_recovered = np.zeros(window_size*(num_col+1))

    i = 0
    for segment in X_recovered.T:
        data_recovered[i*window_size : i*window_size+N_frame ] += segment
        i+= 1
    return data_recovered

def signal_to_noise_ratio(s, s_h):
    return 10*np.log10(np.dot(s, s)/np.dot(s-s_h, s-s_h))

if __name__ == '__main__': # Define parameter
    parser = ArgumentParser(description="De-beep the contaminated sound")
    parser.add_argument('-i', '--infile', help="input wav file") 
    parser.add_argument('-o', '--outfile', help="output wave file") 
    parser.add_argument('-n', '--n', type=int, help="n step shifting to the right") 
    args = parser.parse_args()
    
    sampling_rate, data = scipy.io.wavfile.read(args.infile)
    n = args.n

    # Shifted data
    data_shifted = np.zeros(data.shape[0]).astype(np.int16)
    data_shifted[n:] = data[0:data.shape[0]-n]

    scipy.io.wavfile.write(args.outfile, sampling_rate, data_shifted)
