import deepracin as dr
from scipy import misc
import numpy as np
from skimage import io

import featuresWVT
import featuresFFT

features = []
featuresFFT.fftfeatures(features)
featuresWVT.wvtfeatures(features)
print(features)
