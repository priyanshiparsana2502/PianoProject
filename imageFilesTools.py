# -*- coding: utf-8 -*-
from PIL import Image
import numpy as np


# Returns numpy image at size imageSize*imageSize
def getProcessedData(img, imageSize):
    img = img.resize((imageSize, imageSize), resample=Image.ANTIALIAS)
    imArray = np.asarray(img, dtype=np.uint8)
    newArray = imArray[:,:,0]
    print(np.shape(newArray))
    imgData = newArray.reshape(imageSize, imageSize, 1)
    imgData = imgData / 255.
    return imgData


# Returns numpy image at size imageSize*imageSize
def getImageData(filename, imageSize):
    img = Image.open(filename)
    imgData = getProcessedData(img, imageSize)
    return imgData
