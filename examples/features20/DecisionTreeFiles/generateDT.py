#tree[i][feature, split, lch, rch, ind]
#tree[i][0,       1,      2,   3,  4  ]

import csv

class DecisionTree:
    def __init__(self):
        self.nodes = []
        self.numClasses = None

class Node:
    def __init__(self, id):
        self.id = id
        self.split = None
        self.parent = None
        self.lch = None
        self.rch = None
        self.ind = None

def setNodes(DT, nrNodes):
    # first node has ID 1
    for i in range(1,nrNodes+1):
        DT.nodes.append(Node(i))

def main():

    with open('DT_ISAS.csv', 'rb') as csvfile:
        csvDT = csv.reader(csvfile, delimiter = ',')

        # delete till first value of node field
        for row in csvDT:
            if(row[0] == "node:"):
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
            if(row[0] == "parent:"):
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

        # create DT and set the nr of needed nodes in the DT
        DT = DecisionTree()
        setNodes(DT,nrNodes)

        # skip comma under parent and first "0", because root node has no parent
        next(csvDT)
        next(csvDT)

        #currentCSV = list(csvDT)
        #print(currentCSV[0])

        # we are now at the parent of the first node, which has node id 2 (first "1")
        # the relative index in this loop is the nodeID, we set its parent. we begin with node with ID 2, its parent is the node with the node ID 1
        # set parent nodes. iterate, till we are at "class:" ','
        # start at 1, because nodes[0] is root node, skip it
        for nodeIndex, row in enumerate(csvDT, start=1):
            # only iterate as many times as there are nodes in DT
            # we access the list in DT from (1,nrNodes-1)
            if(row[0] == "class:" or nodeIndex == nrNodes):
                break
            else:
                #print("Row:", row[0])
                DT.nodes[nodeIndex].parent = row[0]
                #print("nodeIndex: ", str(nodeIndex + 1), ", Parent: ", DT.nodes[nodeIndex].parent)




        #for row in csvDT:
        #    print ', '.join(row)



    #for i in range(1,nrNodes):
        #print(DT.nodes[i].id)

if __name__ == "__main__":
   main()
