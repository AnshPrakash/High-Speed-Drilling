import cv2
import numpy as np
import sys

def binarize(img):
  mask = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,101,2)
  return(mask)

def BilateralBlur(img):
  sigI,sigS = 5,5
  kbsize = -1
  blurred = cv2.bilateralFilter(img,kbsize,sigI,sigS)
  blurred = cv2.bilateralFilter(blurred,kbsize,sigI,sigS)
  blurred = cv2.bilateralFilter(blurred,kbsize,sigI,sigS)
  return(blurred)


img = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)

corners  = [[0,0],[0,img.shape[1]-1],[img.shape[0]-1,img.shape[1]-1] , [img.shape[0]-1,0]] 

size = int(max(img.shape[0],img.shape[1])/2.7)

croppedImgs = [0]*4
croppedImgs[0] = img[corners[0][0]:corners[0][0] + size, corners[0][1]:corners[0][1]+size] #Top Left
croppedImgs[1] = img[corners[1][0]:corners[1][0] + size, corners[1][1] - size:corners[1][1]] #Top Right
croppedImgs[2] = img[corners[2][0] - size:corners[2][0], corners[2][1] - size:corners[2][1]] # Bottom right
croppedImgs[3] = img[corners[3][0] - size:corners[3][0], corners[3][1]:corners[3][1] + size] # Bottom Left

lowFimgs = []
for limg in croppedImgs:
  lowFimgs.append(BilateralBlur(limg))


binImgs = []
for bimg in lowFimgs:
  binImgs.append(binarize(bimg))

## check opencv version and construct the detector
is_v2 = cv2.__version__.startswith("2.")
if is_v2:
    detector = cv2.SimpleBlobDetector()
else:
    detector = cv2.SimpleBlobDetector_create()

idx = 0
# Detect blobs.
keypoints = detector.detect(binImgs[idx])
 
lowFimgs[idx] = cv2.cvtColor(lowFimgs[idx],cv2.COLOR_GRAY2BGR)
# Draw detected blobs as red circles.
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
im_with_keypoints = cv2.drawKeypoints(lowFimgs[idx], keypoints, np.array([]), (25,155,185), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
 
# Show keypoints
cv2.imshow("Keypoints", im_with_keypoints)
cv2.waitKey(0)