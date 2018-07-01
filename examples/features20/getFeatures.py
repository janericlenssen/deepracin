import deepracin as dr
from scipy import misc
import numpy as np
from skimage import io

import time
import cProfile as profile

import featuresWVT
import featuresFFT
import DT

def timeFeatures():
    profile.runctx("getFeatures()", globals(), locals())

def decisionTree(F):

    decision = DT.DT(F)
    return decision

def getFeatures():
    features = []
    featuresFFT.fftfeatures(features)
    featuresWVT.wvtfeatures(features)
    concFeatures = np.concatenate((features[0], features[1]), axis=0)
    print(concFeatures)

    decision = decisionTree(concFeatures)
    print("\nDecision: {decision}".replace("{decision}", str(decision)))

if __name__=="__main__":
    #timeFeatures()
    getFeatures()
