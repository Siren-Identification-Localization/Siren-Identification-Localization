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
python train-nb.py <dimensionality reduction model> <classifier model output> <audio training input>
```

For example, we are using the dimensionality reduction model from our previous step:
```
python train-nb.py ambulance.dimred naive_bayes.model dataset/train/**/*.wav
```

### Testing
Now we can use the classifier model to test our test set:
```
python test.py naive_bayes.model datasets/test/**/*.wav
```

### Result
```
[lyulianu@tank]~/Projects/Siren-Localization% python train-dimred.py ambulance.dimred datasets/train/ambulance/*.wav
[lyulianu@tank]~/Projects/Siren-Localization% python train-nb.py ambulance.dimred naive_bayes.model datasets/train/**/*.wav
Training accuracy: 0.9348404255319149
[lyulianu@tank]~/Projects/Siren-Localization% python test.py naive_bayes.model datasets/test/**/*.wav
Testing accuracy: 0.925
```
