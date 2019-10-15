import numpy as np
import pywt
from matplotlib import pyplot as plt
import sys
import cv2

# def hGauss(ksize,theta):
#    ker = np.array(list(range(ksize)))
#    ker - (len(ker)/2.0)


# def anisotropicFilter(sigma,ksize,theta):
#  for theta in np.arange(0, np.pi, np.pi / 16.0): 

def build_filters(lambdda = 6.0,sigma=3.0,gamma = 0.5,psi = 0.2):
 filters = []
 ksize = 31
 for theta in np.arange(0, np.pi, np.pi / 16.0):
   kern = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambdda, gamma, psi, ktype = cv2.CV_32F)
   kern /= 1.5*kern.sum()
   filters.append(kern)
 return filters

def processGabor(img, filters):
 accum = np.zeros_like(img)
 for kern in filters:
  fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
 np.maximum(accum, fimg, accum)
 return accum

img = cv2.imread(sys.argv[1],cv2.IMREAD_GRAYSCALE)

filters = build_filters()
response = processGabor(img,filters)

cv2.imshow("Original",img)
cv2.imshow("Gabor Filter",response)

cv2.waitKey(0)
cv2.destroyAllWindows()
