import librosa
import matplotlib.pyplot as plt
from dtw import dtw
from scipy.spatial.distance import sqeuclidean,mahalanobis
import math

def mfcc(va, vb):
    #Loading audio files
    y1, sr1 = librosa.load(va)
    y2, sr2 = librosa.load(vb)

    mfcc1 = librosa.feature.mfcc(y1,sr1)
    mfcc2 = librosa.feature.mfcc(y2, sr2)

    dist, _, cost, path = dtw(mfcc1.T, mfcc2.T,dist=sqeuclidean)
    dist = math.sqrt(dist/max(mfcc1.shape[0],mfcc2.shape[0]))
    print("The normalized distance between the two : ",dist)
    if dist <= 500:
        return True;
    return False;