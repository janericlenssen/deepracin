from sklearn import tree
from sklearn.tree import _tree
import numpy as np
import re
import copy
import graphviz
from sklearn.model_selection import cross_val_predict, KFold, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
from sklearn.decomposition.pca import PCA

class DecisionTree:
    def __init__(self):
        self.DTcode = ""

DT = DecisionTree()

def trainDT():
    # load train set
    posExamples = open("posFeatures.txt","r").readlines()
    negExamples = open("negFeatures.txt","r").readlines()

    posVector = getData(posExamples)
    negVector = getData(negExamples)

    npPosVec = np.array(posVector)
    npNegVec = np.array(negVector)

    X_train = np.concatenate((npNegVec, npPosVec), axis=0)
    zeros9674 = np.zeros((9674), dtype=int)
    ones9672 = np.full((9672), 1, dtype=int)
    y_train = np.concatenate((zeros9674, ones9672), axis=0)

    # split the data for testing
    #X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=32)

    # train the tree
    clf = tree.DecisionTreeClassifier()
    clf.fit(X_train, y_train)

    # test set
    testFeaturespos = open("testFeaturespos.txt","r").readlines()
    testFeaturesneg = open("testFeaturesneg.txt","r").readlines()

    posTestVector = getData(testFeaturespos)
    negTestVector = getData(testFeaturesneg)

    npTestPosVec = np.array(posTestVector)
    npTestNegVec = np.array(negTestVector)

    X_test = np.concatenate((npTestNegVec, npTestPosVec), axis=0)
    zeros9674_test = np.zeros((9674), dtype=int)
    ones9671 = np.full((9671), 1, dtype=int)
    y_test = np.concatenate((zeros9674_test, ones9671), axis=0)

    #pred = cross_val_predict(clf, X_test, y_test, cv=5)
    #print('Mean of CV:  {:.4%}'.format(accuracy_score(pred, y_test)))

    dt_param = 0.9

    #score_train = accuracy_score(clf.predict(X_train) > dt_param, y_train)
    decision_function = clf.predict(X_test)
    pred = decision_function > dt_param
    score = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred)
    recall = recall_score(y_test, pred)
    average_precision = average_precision_score(y_test, decision_function)

    # print classification performance metrics
    #print('Train score: {:.4%}'.format(score_train))
    print('Test score:  {:.4%}'.format(score))
    print('Precision:   {:.4%}'.format(precision))
    print('Recall:      {:.4%}'.format(recall))
    print('Average pr score: {:.4%}'.format(average_precision))

    # visualize tree
    #dot_data  = tree.export_graphviz(clf, out_file=None, class_names=['0','1'])
    #graph = graphviz.Source(dot_data)
    #graph.render("iris")

    # extract tree code in python from sklearn
    #tree_to_code(clf, ['x0','x1','x2','x3','x4','x5','x6','x7','x8','x9','x10','x11','x12','x13','x14','x15','x16','x17','x18','x19'])

    # show pca curve and pr curve
    pca(clf, X_train, X_test, y_train, y_test, pred)
    pr_curve(y_test, decision_function, average_precision)
    # store trained model
    #file = open("skTree.py","w")
    #file.write(DT.DTcode)
    #file.close()

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

def pca(model, X_train, X_test, y_train, y_test, pred, threshold=0, weights_train=None, weights_test=None):
    pca_model = PCA(2)
    X_train_pca = pca_model.fit_transform(X_train)
    X_test_pca = pca_model.transform(X_test)

    plt.subplot(131)
    plt.suptitle('PCA')
    plt.title('Training data')
    plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], cmap=plt.cm.bwr_r, c=y_train,
                alpha=0.9)
    plt.axis('on')
    plt.subplot(132)
    plt.title('Test data')
    plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], cmap=plt.cm.bwr_r, c=y_test,
                alpha=0.9)
    plt.axis('on')
    plt.subplot(133)
    plt.title('Prediction')
    xx, yy = np.meshgrid(np.linspace(np.min(X_test_pca[:, 0]), np.max(X_test_pca[:, 0]), 50),
                         np.linspace(np.min(X_test_pca[:, 1]), np.max(X_test_pca[:, 1]), 50))
    Z = model.predict(pca_model.inverse_transform(np.c_[xx.ravel(), yy.ravel()])) - threshold
    plt.contourf(xx, yy, Z.reshape(xx.shape), 50, alpha=0.5, cmap=plt.cm.bwr_r, vmin=-1, vmax=1)
    plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], cmap=plt.cm.bwr_r, c=pred,
                alpha=0.9)
    plt.axis('on')
    plt.savefig('pca.png')
    plt.show()

def pr_curve(y_test, y_score, average_precision):

    precision, recall, _ = precision_recall_curve(y_test, y_score)

    plt.step(recall, precision, color='b', alpha=0.5, where='mid')
    plt.fill_between(recall, precision, step='mid', alpha=0.5, color='b')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.15])
    plt.xlim([0.0, 1.15])
    plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(average_precision))
    plt.savefig('pr.png')
    plt.show()

if __name__=="__main__":
    trainDT()
