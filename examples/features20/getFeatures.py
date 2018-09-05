import deepracin as dr
from scipy import misc
import numpy as np
from skimage import io
import os
import glob

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

    # go to folder to test
    scriptPath = os.path.dirname(__file__)
    # set the folder in which the test images are
    destPath = "training/train/pos"
    destPath = os.path.join(scriptPath, destPath)
    #print(destPath)
    # load all images in the test specified folder
    nameList = glob.glob('{path}/*.png'.replace("{path}", destPath))

    first = 0

    for index, imgName in enumerate(nameList):
        #imgName = 'training/train/pos/1.png'
	imgName = "dia64.png"
        features = []
        concFeatures = []
        featuresFFT.fftfeatures(features, imgName, first)
        featuresWVT.wvtfeatures(features, imgName, first)
        concFeatures = np.concatenate((features[0], features[1]), axis=0)
        print(concFeatures)

        decision = decisionTree(concFeatures)
        print("\nDecision: {decision}".replace("{decision}", str(decision)))
        first += 1
        return

if __name__=="__main__":
    #timeFeatures()
    getFeatures()
