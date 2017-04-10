# Identification and Localization of Siren Signals with Beamforming and Non-negative Matrix Factorization

## Team members
- Hankyu Jang (hankjang@iu.edu)
- Leonard Yulianus (lyulianu@iu.edu)
- Sunwoo Kim (kimsunw@iu.edu)

## Datasets
### Preprocessing
Convert audio signal to 16kHz sampling rate and 1 audio channel:
```bash
sox --norm <input> -b 16 <output> rate 16000 channels 1 dither -s
```

## Experiment
### Prerequisite
Since the code are written in Python 3, if you run the code on the campus servers, turn Python 3 module on by running:
```bash
module load python/3.6.0
```

### Dimensionality Reduction Model
First, we need to train a dimensionality reduction model (NMF), run the following command to train the model:
```bash
python train-dimred.py <model output> <audio training input>
```

For example, we are building the model using our training ambulance signals:
```bash
python train-dimred.py ambulance.dimred datasets/train/ambulance/*.wav
```

### Classifier (Naive Bayes)
We train the classifier using the lower dimensional data that will be transformed by our dimensionality reduction model, run the following command:
```bash
python train-nb.py <classifier model output> <audio training input> --dimred <dimensionality reduction model>
```

#### Without Dimensionality reduction
```bash
python train-nb.py naive_bayes.model datasets/train/**/*.wav
```

#### With Dimensionality reduction
```bash
python train-nb.py naive_bayes.model datasets/train/**/*.wav --dimred ambulance.dimred
```

### Testing
Now we can use the classifier model to test our test set:
```bash
python test.py naive_bayes.model datasets/test/**/*.wav
```

### Result
Without dimensionality reduction:
```bash
$ python train-nb.py naive_bayes.model datasets/train/{ambulance,others}/*.wav
Training accuracy: 0.9527027027027027
$ python test.py naive_bayes.model datasets/test/{ambulance,others}/*.wav
Testing accuracy: 0.95
```

With dimensionality reduction:
```bash
$ python train-dimred.py ambulance.dimred datasets/train/ambulance/*.wav
$ python train-nb.py naive_bayes.model datasets/train/{ambulance,others}/*.wav --dimred ambulance.dimred
Training accuracy: 0.9746621621621622
$ python test.py naive_bayes.model datasets/test/{ambulance,others}/*.wav
Testing accuracy: 0.96
```

### Result (on mixed ambulance data)
Without dimensionality reduction:
```bash
$ python train-nb.py naive_bayes.model datasets/train/{ambulance,others}/*.wav
Training accuracy: 0.9527027027027027
$ python test.py naive_bayes.model datasets/test/mixed_ambulance/**/*.wav
Testing accuracy: 0.845
```

With dimensionality reduction:
```bash
$ python train-dimred.py ambulance.dimred datasets/train/ambulance/*.wav
$ python train-nb.py naive_bayes.model datasets/train/{ambulance,others}/*.wav --dimred ambulance.dimred
Training accuracy: 0.9746621621621622
$ python test.py naive_bayes.model datasets/test/mixed_ambulance/**/*.wav
Testing accuracy: 0.89
```

## Visualizing audio signals
We can run the detection against longer audio signals, and see the detection per seconds by running:
```bash
python viz-audio.py <classifier model> <audio signal(s)>
```

For example (show graph to the screen):
```bash
python viz-audio.py naive_bayes.model datasets/{raw/ambulance1.wav,raw/traffic-10.wav,raw_ambulance_mixed/mixed11_16000.wav}
```

For example (save graph to file, useful when running from server):
```bash
python viz-audio.py naive_bayes.model datasets/{raw/ambulance1.wav,raw/traffic-10.wav,raw_ambulance_mixed/mixed11_16000.wav} --save
```

### Some plot results
![ambulance1.wav](plots/ambulance1_plot.png)
![mixed11_16000_plot.png](plots/mixed11_16000_plot.png)
![traffic-10_plot.png](plots/traffic-10_plot.png)
