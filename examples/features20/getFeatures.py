import deepracin as dr
from scipy import misc
import numpy as np
from skimage import io

import time
import cProfile as profile

import featuresWVT
import featuresFFT

def timeFeatures():
    profile.runctx("getFeatures()", globals(), locals())

def decisionTree(F):
    
    return True

def getFeatures():
    features = []
    featuresFFT.fftfeatures(features)
    featuresWVT.wvtfeatures(features)
    concFeatures = np.concatenate((features[0], features[1]), axis=0)
    print(concFeatures)

    decisionTree(concFeatures)

if __name__=="__main__":
    #timeFeatures()
    getFeatures()
