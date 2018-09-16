from sklearn import tree
from sklearn.tree import _tree
import numpy as np
import re
import copy
import graphviz
from sklearn.model_selection import cross_val_predict, KFold, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score

class DecisionTree:
    def __init__(self):
        self.DTcode = ""

DT = DecisionTree()

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

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=32)

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
    #graph = graphviz.Source(dot_data)
    #graph.render("iris")
    tree_to_code(clf, ['x0','x1','x2','x3','x4','x5','x6','x7','x8','x9','x10','x11','x12','x13','x14','x15','x16','x17','x18','x19'])

    file = open("skTree.py","w")
    file.write(DT.DTcode)
    file.close()

    print('Completed.')

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

def tree_to_code(tree, feature_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    DT.DTcode += "def DT({}):".format(", ".join(feature_names))

    def recurse(node, depth):
        indent = "  " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            DT.DTcode += ("\n{}if {} <= {}:".format(indent, name, threshold))
            recurse(tree_.children_left[node], depth + 1)
            DT.DTcode += ("\n{}else:  # if {} > {}".format(indent, name, threshold))
            recurse(tree_.children_right[node], depth + 1)
        else:
            DT.DTcode += ("\n{}return {}".format(indent, np.argmax(tree_.value[node][0])))

    recurse(0, 1)

if __name__=="__main__":
    trainDT()
