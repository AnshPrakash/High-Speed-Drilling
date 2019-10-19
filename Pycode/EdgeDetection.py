import numpy as np
import pywt
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

img = cv2.imread(sys.argv[1],cv2.IMREAD_GRAYSCALE)
mask = cv2.imread(sys.argv[2],cv2.IMREAD_GRAYSCALE)
img = img/255.0
kernel = np.ones((5,5), np.uint8) 
mask = cv2.dilate(mask, kernel, iterations=2) 
mask = mask/255.0

blurred = anisotropicFilter(img,1,3,31)
blurred = blurred*255
blurred = blurred.astype('uint8')
sigI,sigS = 9,9
# ksize = 31
kbsize = -1

blurred = cv2.bilateralFilter(blurred,kbsize,sigI,sigS)
blurred = cv2.bilateralFilter(blurred,kbsize,sigI,sigS)
blurred = cv2.bilateralFilter(blurred,kbsize,sigI,sigS)
blurred = blurred/255.0
kernel = np.ones((9,9))
# gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
# gradient = gradient*255
# gradient = gradient.astype('uint8')
# edges = cv2.Canny(gradient,30,100)

filters = build_filters()
response = processGabor(blurred,filters)
response = response*255
response = response.astype('uint8')
edges = cv2.Canny(response,30,100)
edges = edges/255.0
res = edges*(1-mask)
EdgeMask = edges + mask

# response = processGabor(img,filters)
# print(Gauss2d(5,1,3,20.0))


cv2.imshow("Original",img)
cv2.imshow("Anisotropic Filtering",blurred)
cv2.imshow("Edges",edges)
cv2.imshow("Gabor Filter",response)
cv2.imshow("Mask",mask)
cv2.imshow("ResEdge",res)
cv2.imshow("MaskEdge",EdgeMask)




cv2.waitKey(0)
cv2.destroyAllWindows()
