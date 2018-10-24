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

def decisionTree(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19):

    decision = DT.DT(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19)
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
    #destPath = "training/testposneg/neg"
    destPath = "training/testposneg/neg"
    #destPath = "training/train/pos"
    #destPath = "training/train/neg"
    #destPath = "training/test"

    destPath = os.path.join(scriptPath, destPath)
    destPath = os.path.join(destPath, imageToClassify)
    # load all images in the test specified folder
    #nameList = glob.glob('{path}/*.png'.replace("{path}", destPath))
    # print(nameList)
    # first = 0
    features = []
    concFeatures = []
    featuresFFT.fftfeatures(features, destPath)
    print('**----')
    featuresWVT.wvtfeatures(features, destPath)
    F = np.concatenate((features[0], features[1]), axis=0)

    concFeaturesString = "{imageID} ".replace("{imageID}", imageId)
    concFeaturesString += np.array_str(F)
    concFeaturesString = concFeaturesString.replace('\n', ' ').replace('\r', '').replace('[', '').replace(']', '').replace('  ', ' ')
    concFeaturesString += "\n"

    #file = open("trainingValuesScript/testFeaturespos.txt","a")
    #file = open("trainingValuesScript/testFeatures.txt","a")
    #file = open("trainingValuesScript/posFeatures32.txt","a")
    #file = open("trainingValuesScript/negFeatures32.txt","a")
    file = open("trainingValuesScript/testFeaturesneg32.txt","a")
    file.write(concFeaturesString)
    file.close()

    #print("--")
    #decision = decisionTree(F[0], F[1], F[2], F[3], F[4], F[5], F[6], F[7], F[8], F[9], F[10], F[11], F[12], F[13], F[14], F[15], F[16], F[17], F[18], F[19])
    #print("\nDecision: {decision}".replace("{decision}", str(decision)))
    #print("--")
    #first += 1

if __name__=="__main__":
    #timeFeatures()
    getFeatures()
