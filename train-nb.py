import numpy as np
import scipy.signal as signal
import os.path
from pickle import load, dump
from scipy.io import wavfile
from argparse import ArgumentParser
from sklearn.preprocessing import minmax_scale
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import sklearn.model_selection

def prepare_train_test_using_kfold(k, X, y):
    trainset, testset = [], []
    kf = sklearn.model_selection.KFold(n_splits=k, shuffle=True)
    kf.get_n_splits(X)
    print(kf)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        trainset.append((X_train, y_train))
        testset.append((X_test, y_test))
    return trainset, testset

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

    #################################################################
    # K-fold
    # kfold = 10
    # X = np.array(X)
    # y = np.array(y)
    
    # trainset, testset = prepare_train_test_using_kfold(kfold, X, y)
    # clf_list = []
    # accuracy_list = np.zeros(kfold)
   
    # for i in range(kfold):
        # X_train = trainset[i][0]
        # y_train = trainset[i][1]
        # X_test = testset[i][0]
        # y_test = testset[i][1]
        # clf = GaussianNB()
        # pred = clf.fit(X_train, y_train).predict(X_test)
        # accuracy = accuracy_score(y_test, pred)
        # print('Accuracy fold{}: {}'.format(i+1, accuracy))
        # clf_list.append(clf)
        # accuracy_list[i] = accuracy

    # classifier = clf_list[accuracy_list.argmax()]
    #################################################################

    classifier = GaussianNB()
    y_hat = classifier.fit(X, y).predict(X)
    print('Training accuracy: {}'.format(accuracy_score(y, y_hat)))

    # Save classifier
    with open(args.output, 'wb') as f:
        dump([dimred, classifier], f)
