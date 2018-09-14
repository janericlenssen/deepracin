from sklearn import tree
import numpy as np
import re
import copy
import graphviz
from sklearn.model_selection import cross_val_predict, KFold, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score


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

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=3422, )

    clf = tree.DecisionTreeClassifier()
    clf.fit(X_train, y_train)

    pred = cross_val_predict(clf, X_test, y_test, cv=5)
    print('Mean of CV:  {:.4%}'.format(accuracy_score(pred, y_test)))

    dt_param = 0.9

    score_train = accuracy_score(clf.predict(X_train) > dt_param, y_train)
    decision_function = clf.predict(X_test)
    pred = decision_function > dt_param
    score = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred)
    recall = recall_score(y_test, pred)

    print('Train score: {:.4%}'.format(score_train))
    print('Test score:  {:.4%}'.format(score))
    print('Precision:   {:.4%}'.format(precision))
    print('Recall:      {:.4%}'.format(recall))

    dot_data  = tree.export_graphviz(clf, out_file=None, class_names=['0','1'])
    graph = graphviz.Source(dot_data)
    graph.render("iris")
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
