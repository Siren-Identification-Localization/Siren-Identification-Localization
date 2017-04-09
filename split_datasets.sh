#!/bin/bash

# Put 20 sounds for each ambulance and others classes randomly into test folders
FILES=$(find datasets/ambulance/ -iname *.wav | shuf | head -20)
cp $FILES datasets/test/ambulance/
FILES=$(find datasets/others/ -iname *.wav | shuf | head -20)
cp $FILES datasets/test/others/

# Put the rest into training folders
cp datasets/ambulance/*.wav datasets/train/ambulance/
cp datasets/others/*.wav datasets/train/others/
cd datasets/test/ambulance/
for FILE in *.wav; do rm ../../train/ambulance/$FILE; done
cd ../others/
for FILE in *.wav; do rm ../../train/others/$FILE; done
