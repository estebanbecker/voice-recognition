import numpy
import os
import librosa
from sklearn.mixture import GaussianMixture

train_data = []
train_labels = []

path="Voice_Dataset/"

for file in os.listdir(path+"train/"):
    if file.endswith(".wav"):
        y, sr = librosa.load(path + "train/" + file)
        mfcc = librosa.feature.mfcc(y=y, sr=sr)
        train_data.append(mfcc)
        train_labels.append(file.split("-")[0])