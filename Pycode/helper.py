import numpy as np
from matplotlib import pyplot as plt
from skimage.transform import rotate
import sys
import cv2

def Gauss1D(ksize,sigma):
  ker = np.array(list(range(ksize)))
  ker = ker - int(len(ker)/2.0)
  ker = np.exp(-ker**2/(2*sigma*sigma))
  ker = ker/np.sum(ker)
  return(ker)

def Gauss2d(ksize,sigma1,sigma2,theta):
  ker1 = Gauss1D(ksize,sigma1)
  ker2 = Gauss1D(ksize,sigma2)
  ker2D_x = np.zeros((ksize,ksize))
  ker2D_y = np.zeros((ksize,ksize))
  ker2D_x[int(ksize/2)] = ker1
  ker2D_y[:,int(ksize/2)] = ker2
  ker = cv2.filter2D(ker2D_x,cv2.CV_64F,ker2D_y)
  ker = rotate(ker, theta, resize=True)
  ker = ker/np.sum(ker)
  return(ker)


def anisotropicFilter(img,sigma1,sigma2,ksize):
  MaxAngle = 360.0
  # MaxAngle = np.pi
  accum = np.zeros_like(img)
  for theta in np.arange(0, MaxAngle, MaxAngle / 16.0):
    ker = Gauss2d(ksize,sigma1,sigma2,theta)
    fimg = cv2.filter2D(img,cv2.CV_64F,ker)
    np.maximum(accum, fimg, accum)
  return(accum)



def build_filters(lambdda = 7.0,sigma=3.0,gamma = 1.0,psi = 0.0):
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
    fimg = cv2.filter2D(img, cv2.CV_64F, kern)
    np.maximum(accum, fimg, accum)
  return accum
