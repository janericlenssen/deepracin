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
import argparse

def timeFeatures():
    profile.runctx("getFeatures()", globals(), locals())

def decisionTree(F):

    decision = DT.DT(F)
    return decision

def getFeatures():
    # get the training image from command line parameter
    parser = argparse.ArgumentParser(description='Image file name')
    parser.add_argument('--imageName', type=str, default=None, help='No help.')
    parser.add_argument('--id', type=str, default=None, help='No help.')
    args = parser.parse_args()
    imageToClassify = args.imageName
    imageId = args.id
    # go to folder to test
    scriptPath = os.path.dirname(__file__)
    # set the folder in which the test images are
    destPath = "training/train/neg"
    #destPath = "training/train/pos"
    destPath = os.path.join(scriptPath, destPath)
    destPath = os.path.join(destPath, imageToClassify)
    # load all images in the test specified folder
    #nameList = glob.glob('{path}/*.png'.replace("{path}", destPath))
    # print(nameList)
    # first = 0
    features = []
    concFeatures = []
    featuresFFT.fftfeatures(features, destPath)
    featuresWVT.wvtfeatures(features, destPath)
    concFeatures = np.concatenate((features[0], features[1]), axis=0)

    concFeaturesString = "{imageID} ".replace("{imageID}", imageId)
    concFeaturesString += np.array_str(concFeatures)
    concFeaturesString = concFeaturesString.replace('\n', ' ').replace('\r', '').replace('[', '').replace(']', '').replace('  ', ' ')
    concFeaturesString += "\n"

    file = open("trainingValuesScript/negFeatures.txt","a")
    #file = open("trainingValuesScript/posFeatures.txt","a")
    file.write(concFeaturesString)
    file.close()

    #decision = decisionTree(concFeatures)
    #print("\nDecision: {decision}".replace("{decision}", str(decision)))
    #first += 1

if __name__=="__main__":
    #timeFeatures()
    getFeatures()
