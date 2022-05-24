import numpy as np
import os
import librosa
from sklearn.mixture import GaussianMixture

trained_gaussian=[]
train_labels = []
n_mfcc = 45
n_components = 39

path="Voice_Dataset/"

for file in os.listdir(path+"train/"):
    if file.endswith(".wav"):
        y, sr = librosa.load(path + "train/" + file)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        
        mfcc = np.transpose(mfcc)


        gaussian = GaussianMixture(n_components=n_components, random_state=3)
        gaussian = gaussian.fit(mfcc)

        trained_gaussian.append(gaussian)
        train_labels.append(file[1])



test_data = []
test_labels = []
real_labels = []
predicted_labels = []

for file in os.listdir(path+"test/"):
    if file.endswith(".wav"):
        y, sr = librosa.load(path + "test/" + file)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        
        mfcc = np.transpose(mfcc)

        
        real_labels.append(file[1])

        prediction=[]
        for gaussian in trained_gaussian:
            prediction.append(gaussian.score(mfcc))

        
        predicted_labels.append(train_labels[prediction.index(max(prediction))])

print("Real labels:")
print(real_labels)
print("predicted labels:")
print(predicted_labels)

i=0
count=0
while(i < len(real_labels)):
    if real_labels[i] == predicted_labels [i]:
        count+=1
    
    i+=1

print("Precision: "+str(count/len(real_labels))+"%")
