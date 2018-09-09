from sklearn import tree
import numpy as np
import re
import copy



def trainDT():
    posExamples = open("posFeatures.txt","r").readlines()
    negExamples = open("negFeatures.txt","r").readlines()

    posVector = getData(posExamples)
    negVector = getData(negExamples)

    npPosVec = np.array(posVector)
    npNegVec = np.array(negVector)

    X = np.concatenate((npNegVec, npPosVec), axis=0)
    zeros9674 = np.zeros((9674), dtype=int)
    ones9672 = np.full((9672), 1, dtype=int)
    Y = np.concatenate((zeros9674, ones9672), axis=0)

    clf = tree.DecisionTreeClassifier()
    clf.fit(X, Y)

    print('Training completed.')

def getData(examples):
    allFeatures = []
    # store every feature vector in numpy array
    for index, line in enumerate(examples):
        # empty array where current floats are stored
        arrayLine = []

        # there are no features in the first line
        if (index == 0):
            continue

        # remove image IDs
        line = re.sub(r'^\d* ', '', line)
        # remove newlines at end
        line = re.sub(r'\n', '', line)
        # print(line)

        # take all values in the line and
        tempLine = copy.copy(line)
        for i in range(20):
            # no space follows last decimal
            if (i == 19):
                floatVal = float(tempLine)
                #print(stringVal)
                arrayLine.append(floatVal)
            else:
                stringVal = re.search('^\S* ', tempLine).group(0)
                stringVal = re.sub(r' ', '', stringVal)
                floatVal = float(stringVal)
                #print(stringVal)

                arrayLine.append(floatVal)
                # remove the already added float from tempLine
                tempLine = re.sub('^\S* ', '', tempLine)

        # add new array to allFeatures
        allFeatures.append(arrayLine)
        #print(arrayLine)
        #if (index == 1):
        #    break
    return allFeatures

if __name__=="__main__":
    trainDT()
