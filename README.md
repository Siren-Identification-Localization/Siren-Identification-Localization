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
