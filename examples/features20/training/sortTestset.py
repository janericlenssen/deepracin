import csv
from shutil import copyfile
import os

def readCSV():

    with open('annotations_test.csv', 'rb') as csvfile:
        csvDT = csv.reader(csvfile, delimiter = '\n')
        counterP = 0
        destpath = 'testposneg/'
        # read csv row
        for row in csvDT:
            print(counterP)
            pathAndD = row[0]
            decision = int(pathAndD[-1:])
            pathFile = pathAndD[1:-2]
            filename = pathFile[5:]
            print(pathFile, decision, filename)

            # if 1, copy in pos directoty
            if decision == 1:
                destpathPos = os.path.join(destpath, 'pos', filename)
                print(pathFile, destpathPos)
                copyfile(pathFile, destpathPos)
            elif decision == 0:
                destpathNeg = os.path.join(destpath, 'neg', filename)
                print(pathFile, destpathNeg)
                copyfile(pathFile, destpathNeg)
            else:
                print('error!')
                break

            counterP += 1
            #if counterP == 120:
            #    break


if __name__ == "__main__":
   readCSV()
