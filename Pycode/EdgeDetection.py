import numpy as np
from matplotlib import pyplot as plt
from skimage.transform import rotate
import sys
import cv2
from helper import *


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


############ 
res = res*255
res = res.astype('uint8')

# lines = cv2.HoughLines(res,1,np.pi/180,50)
# for line in lines:
#   for rho,theta in line:
#     a = np.cos(theta)
#     b = np.sin(theta)
#     x0 = a*rho
#     y0 = b*rho
#     x1 = int(x0 + 1000*(-b))
#     y1 = int(y0 + 1000*(a))
#     x2 = int(x0 - 1000*(-b))
#     y2 = int(y0 - 1000*(a))
#     cv2.line(img,(x1,y1),(x2,y2),(0,0,0),1)


minLineLength = 1
maxLineGap = 1
lines = cv2.HoughLinesP(res,1,np.pi/180,25,minLineLength,maxLineGap)
for line in lines:
  for x1,y1,x2,y2 in line:
    cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)

##########
# response = processGabor(img,filters)
# print(Gauss2d(5,1,3,20.0))


cv2.imshow("Original",img)
# cv2.imshow("Anisotropic Filtering",blurred)
# cv2.imshow("Edges",edges)
# cv2.imshow("Gabor Filter",response)
# cv2.imshow("Mask",mask)
# cv2.imshow("MaskEdge",EdgeMask)
cv2.imshow("ResEdge",res)




cv2.waitKey(0)
cv2.destroyAllWindows()
