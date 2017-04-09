# Identification and Localization of Siren Signals with Beamforming and Non-negative Matrix Factorization

## Team members
- Hankyu Jang (hankjang@iu.edu)
- Leonard Yulianus (lyulianu@iu.edu)
- Sunwoo Kim (kimsunw@iu.edu)

## Datasets
### Preprocessing
Convert audio signal to 16kHz sampling rate and 1 audio channel:
```
sox --norm <input> -b 16 <output> rate 16000 channels 1 dither -s
```

## Experiment
### Prerequisite
Since the code are written in Python 3, if you run the code on the campus servers, turn Python 3 module on by running:
```
module load python/3.6.0
```

### Dimensionality Reduction Model
First, we need to train a dimensionality reduction model (NMF), run the following command to train the model:
```
python train-dimred.py <model output> <audio training input>
```

For example, we are building the model using our training ambulance signals:
```
python train-dimred.py ambulance.dimred datasets/train/ambulance/*.wav
```

### Classifier (Naive Bayes)
We train the classifier using the lower dimensional data that will be transformed by our dimensionality reduction model, run the following command:
```
python train-nb.py <classifier model output> <audio training input> --dimred <dimensionality reduction model>
```

#### Without Dimensionality reduction
```
python train-nb.py naive_bayes.model datasets/train/**/*.wav
```

#### With Dimensionality reduction
```
python train-nb.py naive_bayes.model datasets/train/**/*.wav --dimred ambulance.dimred
```

### Testing
Now we can use the classifier model to test our test set:
```
python test.py naive_bayes.model datasets/test/**/*.wav
```

### Result
Without dimensionality reduction:
```
$ python train-nb.py naive_bayes.model datasets/train/{ambulance,others}/*.wav
Training accuracy: 0.9527027027027027
$ python test.py naive_bayes.model datasets/test/**/*.wav
Testing accuracy: 0.95
```

With dimensionality reduction:
```
$ python train-dimred.py ambulance.dimred datasets/train/ambulance/*.wav
$ python train-nb.py naive_bayes.model datasets/train/{ambulance,others}/*.wav --dimred ambulance.dimred
Training accuracy: 0.9324324324324325
$ python test.py naive_bayes.model datasets/test/**/*.wav
Testing accuracy: 0.92
```
