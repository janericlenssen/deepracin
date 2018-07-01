#tree[i][feature, split, lch, rch, ind]
#tree[i][0,       1,      2,   3,  4  ]

import csv

class DecisionTree:
    def __init__(self):
        self.nodes = []
        self.numClasses = None

    def setNodes(self, nrNodes):
        # first node has ID 1
        for i in range(1,nrNodes+1):
            self.nodes.append(Node(i))

class Node:
    def __init__(self, id):
        self.id = id
        self.cut = None
        self.parent = None
        self.lch = None
        self.rch = None
        self.ind = None
        self.classification = None
        self.featureNr = None


def main():
    DT = DecisionTree()
    readCSV(DT)
    generateDT(DT) #to C, then exec from lib

def generateDT(DT):
    return

def readCSV(DT):

    with open('DT_ISAS.csv', 'rb') as csvfile:
        csvDT = csv.reader(csvfile, delimiter = ',')

        # delete till first value of node field
        for row in csvDT:
            if (row[0] == "node:"):
                #print(row[0])
                next(csvDT)
                break
            else:
                next(csvDT)

        #currentCSV = list(csvDT)
        #print(currentCSV[0])

        # now we are in the row which holds the first value (id) for the nodes

        # skip the number of rows till "parent" row, and count skips
        # we count the number of needed nodes
        nrNodes = 0
        for row in csvDT:
            if (row[0] == "parent:"):
                #print(row)
                #next(csvDT)
                break
            else:
                #print(row)
                #next(csvDT)
                nrNodes += 1

        # we skip one more than there are nodes
        nrNodes -= 1
        #print(nrNodes)

        # set the nr of needed nodes in the DT
        DT.setNodes(nrNodes)

        # skip comma under parent and first "0", because root node has no parent
        next(csvDT)
        next(csvDT)

        # we are now at the parent of the first node, which has node id 2 (first "1")
        # the relative index in this loop is the nodeID, we set its parent. we begin with node with ID 2, its parent is the node with the node ID 1
        # set parent nodes. iterate, till we are at "class:" ','
        # start at 1, because nodes[0] is root node, skip it
        for nodeIndex, row in enumerate(csvDT, start=1):
            # only iterate as many times as there are nodes in DT
            # we access the list in DT from (1,nrNodes-1)
            if (nodeIndex == nrNodes):
                break
            else:
                #print("Row:", row[0])
                DT.nodes[nodeIndex].parent = row[0]
                #print("nodeIndex: ", str(nodeIndex + 1), ", Parent: ", DT.nodes[nodeIndex].parent)

        # we are in the "class:" row now, skip "class:" and ", " row
        next(csvDT)
        next(csvDT)

        # add the classes
        for nodeIndex, row in enumerate(csvDT):
            if (nodeIndex == (nrNodes)):
                break
            else:
                # use 1 as virus, 0 as no virus (0 instead of 2), and 255 as an error case
                row0 = [1 if row[0]=="1" else 0 if row[0]=="2" else 255]
                DT.nodes[nodeIndex].classification = row0
                #print(DT.nodes[nodeIndex].classification)

        # we are now at "var:"
        next(csvDT)
        next(csvDT)

        for nodeIndex, row in enumerate(csvDT):
            if (nodeIndex == (nrNodes)):
                break
            else:
                DT.nodes[nodeIndex].featureNr = row[0]
                #print(DT.nodes[nodeIndex].featureNr)

        # we are now at "cut:"
        next(csvDT)
        next(csvDT)

        for nodeIndex, row in enumerate(csvDT):
            if (nodeIndex == (nrNodes)):
                break
            else:
                DT.nodes[nodeIndex].cut = row[0]
                #print(DT.nodes[nodeIndex].cut)

        # we are now at "children:"
        next(csvDT)
        next(csvDT)

        for nodeIndex, row in enumerate(csvDT):
            if (nodeIndex == (nrNodes)):
                break
            else:
                DT.nodes[nodeIndex].lch = row[0]
                DT.nodes[nodeIndex].rch = row[1]
                #print(DT.nodes[nodeIndex].lch, DT.nodes[nodeIndex].rch)

        next(csvDT)
        next(csvDT)

        #print(row)
        currentCSV = list(csvDT)
        DT.numClasses = currentCSV[0][0]
        
        # print all nodes
        """
        for i in range(nrNodes):
            node = DT.nodes[i]
            print("ID:  ", node.id  , \
            "Parent:    ", node.parent  , \
            "Class: ", node.classification  , \
            "Var:   ", node.featureNr   , \
            "Cut:   ", node.cut , \
            "Children:  ", node.lch , node.rch  )
        """
        #for row in csvDT:
        #    print ', '.join(row)



    #for i in range(1,nrNodes):
        #print(DT.nodes[i].id)

if __name__ == "__main__":
   main()
