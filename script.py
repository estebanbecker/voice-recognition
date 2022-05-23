import numpy as np
import os
import librosa
from sklearn.mixture import GaussianMixture

train_data = []
train_labels = []

path="Voice_Dataset/"

for file in os.listdir(path+"train/"):
    if file.endswith(".wav"):
        y, sr = librosa.load(path + "train/" + file)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=200)
        
        mfcc = np.transpose(mfcc)

        for data in mfcc:
            train_data.append(data)
            train_labels.append(file[1])

gaussian = GaussianMixture(n_components=482,random_state=3)
gaussian = gaussian.fit(train_data)

test_data = []
test_labels = []
real_labels = []
predicted_labels = []

for file in os.listdir(path+"test/"):
    if file.endswith(".wav"):
        y, sr = librosa.load(path + "test/" + file)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=200)
        
        mfcc = np.transpose(mfcc)

        for data in mfcc:
            test_data.append(data)
            test_labels.append(file[1])
        
        prediction = gaussian.predict(test_data)
        print(prediction)
        count=[0]*9
        for i in prediction:
            count[int(train_labels[int(i)])-1] += 1
        print(count)
        
        predicted_labels.append(count.index(max(count)))
        real_labels.append(file[1])

print(real_labels)
print(predicted_labels)