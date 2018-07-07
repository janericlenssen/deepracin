import cv2
import os
import glob
from PIL import Image

# pip install opencv-python
# copy this script into the folder with images that should be zeropadded and execute with python3
# size of the image to which it should be padded to
padTo = 64
bottom = 16
top = 0
left = 0
right = 16

def main():

    # create directory for zeropadded images
    scriptPath = os.path.dirname(__file__)
    destPath = "zeroPadded"
    destPath = os.path.join(scriptPath, destPath)
    if not os.path.exists(destPath):
        os.makedirs(destPath)

    imgPath = scriptPath

    # get all the files with names
    nameList = glob.glob('*.png')
    #print(nameList)

    for imgName in nameList:
        dirImgName = os.path.join(imgPath, imgName)
        dirPaddedImgName = os.path.join(destPath, imgName)
        #print(dirPaddedImgName)
        loadedImg = cv2.imread(dirImgName, 0)
        #cv2.imshow(imgName, loadedImg)
        #cv2.waitKey(0)
        paddedImg = cv2.copyMakeBorder(loadedImg, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
        #cv2.imshow(imgName, paddedImg)
        cv2.imwrite(dirPaddedImgName, paddedImg)
        #cv2.waitKey(0)

if __name__=="__main__":
    main()
